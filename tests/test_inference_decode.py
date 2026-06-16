import importlib
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal local envs.
    torch = None


def _load_decode():
    inference = importlib.import_module("inference")
    return inference._decode_3angle_prediction


def _prediction(out_dim, azimuth, polar_bin, rotation_bin):
    pred = torch.zeros(1, out_dim)
    pred[0, azimuth] = 10.0
    pred[0, 360 + polar_bin] = 11.0
    pred[0, 360 + 180 + rotation_bin] = 12.0
    pred[0, -2] = 8.0
    pred[0, -1] = -8.0
    return pred


@unittest.skipIf(torch is None, "torch is not installed")
class DecodePredictionTest(unittest.TestCase):
    def test_decode_prediction_supports_current_and_legacy_heads(self):
        decode = _load_decode()

        for out_dim, rotation_bin, rotation_offset in [(902, 181, 180), (722, 91, 90)]:
            with self.subTest(out_dim=out_dim):
                azimuth, polar_bin = 12, 95
                pred = _prediction(out_dim, azimuth, polar_bin, rotation_bin)
                if hasattr(torch, "npu") and torch.npu.is_available():
                    pred = pred.to("npu:0")

                az, polar, rotation, confidence, offset = decode(pred)

                self.assertEqual(int(az[0]), azimuth)
                self.assertEqual(int(polar[0] - 90), polar_bin - 90)
                self.assertEqual(int(rotation[0] - offset), rotation_bin - rotation_offset)
                self.assertEqual(offset, rotation_offset)
                self.assertGreater(float(confidence[0]), 0.999)

    def test_decode_prediction_rejects_unknown_head_shape(self):
        decode = _load_decode()
        pred = torch.zeros(1, 800)
        if hasattr(torch, "npu") and torch.npu.is_available():
            pred = pred.to("npu:0")

        with self.assertRaisesRegex(ValueError, "Unsupported prediction dimension"):
            decode(pred)


if __name__ == "__main__":
    unittest.main()
