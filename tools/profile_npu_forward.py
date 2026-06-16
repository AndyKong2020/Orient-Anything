#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu
from PIL import Image
from transformers import AutoImageProcessor

from inference import _decode_3angle_prediction
from paths import DINO_LARGE
from vision_tower import DINOv2_MLP


def create_profiler(save_path: str, active: int):
    os.makedirs(save_path, exist_ok=True)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU,
            torch_npu.profiler.ProfilerActivity.CPU,
        ],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        experimental_config=experimental_config,
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=0, active=active, repeat=1, skip_first=0
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(save_path),
    )


def find_profiler_output(save_path: Path) -> Path:
    outputs = sorted(save_path.glob("**/ASCEND_PROFILER_OUTPUT"))
    if not outputs:
        raise FileNotFoundError(f"ASCEND_PROFILER_OUTPUT not found under {save_path}")
    return outputs[-1]


def load_model(device: str, ckpt_path: str) -> DINOv2_MLP:
    model = DINOv2_MLP(
        dino_mode="large",
        in_dim=1024,
        out_dim=360 + 180 + 360 + 2,
        evaluate=True,
        mask_dino=False,
        frozen_back=False,
    )
    model.eval()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(device)


def prepare_input(image_path: str, device: str):
    cache_dir = os.environ.get("ORIENT_CACHE_DIR") or os.environ.get("HF_HOME") or "./"
    processor = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir=cache_dir)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image)
    pixel_values = torch.from_numpy(np.array(inputs["pixel_values"])).to(device)
    return {"pixel_values": pixel_values}


def run_forward(model, inputs):
    with torch.no_grad():
        out = model(inputs)
    torch.npu.synchronize()
    return out


def summarize_output(output_dir: Path):
    kernel_details = output_dir / "kernel_details.csv"
    trace_view = output_dir / "trace_view.json"
    op_statistic = output_dir / "op_statistic.csv"
    api_statistic = output_dir / "api_statistic.csv"

    with kernel_details.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row_count = sum(1 for _ in reader)

    try:
        json.load(trace_view.open())
        trace_valid = True
    except Exception:
        trace_valid = False

    return {
        "profiler_output": str(output_dir),
        "kernel_details_rows": row_count,
        "kernel_details_cols": len(header),
        "trace_view_valid_json": trace_valid,
        "op_statistic_exists": op_statistic.exists(),
        "api_statistic_exists": api_statistic.exists(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="assets/demo.png")
    parser.add_argument("--ckpt", default=os.environ.get("ORIENT_CKPT_PATH"))
    parser.add_argument("--save-path", default="/tmp/orient_anything_profile")
    parser.add_argument("--active", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    if not args.ckpt:
        raise SystemExit("Set --ckpt or ORIENT_CKPT_PATH to the Orient-Anything weight file")
    if not torch.npu.is_available():
        raise SystemExit("NPU is not available")

    device = "npu:0"
    torch.npu.set_device(device)
    save_path = Path(args.save_path)
    model = load_model(device, args.ckpt)
    inputs = prepare_input(args.image, device)

    for _ in range(args.warmup):
        run_forward(model, inputs)

    step_ms = []
    with create_profiler(str(save_path), args.active) as prof:
        for _ in range(args.active):
            t0 = time.perf_counter()
            out = run_forward(model, inputs)
            step_ms.append((time.perf_counter() - t0) * 1000)
            prof.step()
        prof.step()

    preds = _decode_3angle_prediction(out)
    output_dir = find_profiler_output(save_path)
    summary = summarize_output(output_dir)
    summary.update(
        {
            "active_steps": args.active,
            "step_ms": [round(x, 3) for x in step_ms],
            "mean_step_ms": round(sum(step_ms) / len(step_ms), 3),
            "prediction": {
                "azimuth": float(preds[0][0]),
                "polar": float(preds[1][0] - 90),
                "rotation": float(preds[2][0] - preds[4]),
                "confidence": round(float(preds[3][0]), 6),
            },
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
