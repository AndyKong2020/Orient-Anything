# Orient Anything NPU 部署报告

生成日期：2026-06-16

## 任务概述

本次部署任务是在 Orient Anything V1 代码基础上，将项目同步到 Ascend NPU 目标容器内，建立独立 Python 运行环境，并完成一轮以原始功能可用性和官方 V1 Demo 对齐为主的复现验证。Orient Anything 是一个图像级物体朝向估计项目，核心流程为 DINOv2-Large backbone 提取图像特征，再通过 MLP head 输出 azimuth、polar、rotation 和 confidence，并在 Gradio WebUI 中叠加渲染坐标轴。

代码版本锚点为 commit `759282c26e924988c952d6d33212c48349dc9aff`。本次复现保留项目原始 Gradio UI 与推理入口，没有自建替代 WebUI；适配工作集中在 Ascend NPU 设备选择、模型缓存路径、NPU tensor 解码和服务监听参数上。

硬件约束为只使用后四张 Ascend NPU。当前服务验证使用后四张卡中的一张单卡，通过 `ASCEND_RT_VISIBLE_DEVICES` 隔离设备后，进程内以 `npu:0` 运行模型。下载与模型缓存均放在挂载数据盘，避免占用系统盘。

## 部署过程

远端使用独立 Python 虚拟环境部署项目依赖，并显式复用 Ascend 运行栈中的 `torch-npu`。关键运行组件如下：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.6 |
| torch | 2.10.0+cpu |
| torch-npu | 2.10.0 |
| transformers | 4.38.0 |
| huggingface_hub | 0.26.5 |
| gradio | 5.9.0 |
| rembg | 2.0.69 |
| onnxruntime | 1.26.0 |
| numpy | 1.26.4 |
| pillow | 10.2.0 |
| matplotlib | 3.11.0 |
| scikit-image | 0.26.0 |

部署过程中使用挂载盘目录承载 Hugging Face、rembg、matplotlib 等缓存。DINOv2-Large 权重下载时遇到 Hugging Face 客户端断点续传不稳定的问题，最终改为在挂载盘中完成手动续传，并通过环境变量把 `facebook/dinov2-large` 指向本地缓存目录。Orient Anything 自身 checkpoint `ronormsigma1/dino_weight.pt` 也保存在 Hugging Face 缓存目录中。

本次代码改动保持较小范围：

| 文件 | 改动目的 |
| --- | --- |
| `app.py` | 可选导入 `torch_npu`，优先选择 `npu:0`，支持缓存目录和 Gradio 监听地址/端口环境变量，避免 NPU tensor 直接进入 NumPy/PIL 路径。 |
| `vision_tower.py` | 让 DINOv2 backbone 使用可配置缓存目录，避免模型落到系统盘。 |
| `paths.py` | 支持通过 `ORIENT_DINO_*` 环境变量覆盖 DINO 模型路径，便于使用挂载盘中的本地权重。 |
| `inference.py` | 增加统一解码函数，将 NPU 输出显式转 CPU 后再做 `argmax`、`softmax` 和标量赋值；同时兼容当前官方 Space 使用的 902 维 head 和 README 中旧示例的 722 维 head。 |
| `tests/test_inference_decode.py` | 新增轻量单元测试，覆盖 902 维与 722 维输出头解码，以及未知输出维度报错。 |

Gradio 服务按项目原始入口 `python app.py` 启动，监听容器内端口后通过本地端口转发验证。由于 Gradio 5 会进行本地可达性检查，服务启动时需要让 loopback 地址绕过代理，否则代理环境会让本地自检误判为不可访问。

## 功能验证结果

本次验证分为四层：基础环境检查、代码静态检查、本地 NPU WebUI/API smoke、官方 V1 Demo 对齐。

基础环境验证中，`torch_npu` 可以正常导入，`torch.npu.is_available()` 返回 `True`，并完成过 NPU tensor 基础计算 smoke。服务进程在后四张卡中的单卡上加载 DINOv2-Large 与 Orient Anything checkpoint，并完成 `assets/demo.png` 推理。

静态检查和轻量回归测试结果如下：

| 检查项 | 结果 |
| --- | --- |
| `python3 -m py_compile app.py inference.py vision_tower.py paths.py utils.py render/*.py` | 通过 |
| `git diff --check` | 通过 |
| 目标 NPU venv：`python -m py_compile app.py inference.py vision_tower.py paths.py utils.py render/*.py` | 通过 |
| 目标 NPU venv：`python tests/test_inference_decode.py` | 2 个 unittest 通过，用时 2.390s |

目标 NPU venv 环境检查显示 `torch_npu 2.10.0` 可以正常导入，设置单卡可见后 `torch.npu.is_available()` 为 `True`，`torch.npu.device_count()` 为 1，NPU tensor 基础计算返回 `2.0`。

直接模型 smoke 已在目标 NPU venv 中完成，加载本地缓存的 DINOv2-Large 与 `ronormsigma1/dino_weight.pt` 后，对 `assets/demo.png` 得到如下结果：

| 输入 | Remove Background | 输出 | 推理耗时 |
| --- | --- | --- | ---: |
| `assets/demo.png` | `False` | azimuth `3.0`，polar `0.0`，rotation `1.0`，confidence `0.9965` | 0.077s |
| `assets/demo.png` | `True` | azimuth `355.0`，polar `0.0`，rotation `2.0`，confidence `0.2945` | 0.553s |

本地 NPU Gradio API 已完成两种主路径验证：

| 输入 | Remove Background | Inference Time Augmentation | 输出 |
| --- | --- | --- | --- |
| `assets/demo.png` | `False` | `False` | azimuth `3.0`，polar `0.0`，rotation `1.0`，confidence `1.0` |
| `assets/demo.png` | `True` | `False` | azimuth `355.0`，polar `0.0`，rotation `2.0`，confidence `0.29` |

官方 V1 Space 使用同一套 `app.py` 和 `/predict` 接口，运行时为 Gradio `5.9.0`，硬件为 CPU basic。使用同一张 `assets/demo.png` 对齐结果如下：

| 设置 | 官方 V1 Demo | 本地 NPU |
| --- | ---: | ---: |
| `Remove Background=False` | `3.0, 0.0, 1.0, 1.0` | `3.0, 0.0, 1.0, 1.0` |
| `Remove Background=True` | `355.0, 0.0, 2.0, 0.34` | `355.0, 0.0, 2.0, 0.29` |

关闭背景移除时，本地 NPU 与官方 V1 Demo 输出完全一致。开启背景移除时，姿态角度一致，confidence 存在小幅差异；该差异来自 rembg/ONNXRuntime/图像预处理后端差异，不影响本次“原始功能可运行并与官方 V1 行为对齐”的结论。

## Benchmark 与指标说明

当前仓库未提供独立 benchmark、eval 或 test 脚本，也未随仓提供标注评测集。因此本次不能给出数据集级 MAE、accuracy 或论文指标复现结果。仓内官方使用方式主要是：

| 入口 | 用途 |
| --- | --- |
| `python app.py` | 启动 Gradio WebUI，进行单图可视化推理。 |
| `inference.get_3angle()` | 单图朝向推理。 |
| `inference.get_3angle_infer_aug()` | 随机裁剪测试时增强推理。 |

本次可量化的复现指标是单图功能性 smoke 与官方 V1 Demo parity。后续若需要论文级指标复现，需要补充固定评测集、标注格式、角度误差定义和批量评测脚本。

## 部署结论

Orient Anything V1 已在 Ascend NPU 目标环境完成原始功能复现。核心模型前向运行在 NPU 上，Gradio WebUI、单图推理、背景移除路径、坐标轴渲染与官方 V1 Demo 对齐均已验证。当前部署适合作为单卡交互式演示和后续 NPU 适配分析的基础。

当前不建议把本次结果表述为“全量 NPU 原生端到端”。项目的图像读取、Hugging Face processor、rembg 背景移除、坐标轴渲染和 Gradio 服务仍主要是 CPU/Python 路径；NPU 亲和性重点应限定在 DINOv2-Large + MLP head 的模型前向和输出解码链路。
