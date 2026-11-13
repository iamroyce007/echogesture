"""Convert checkpoints/model.pt -> ONNX FP32 -> ONNX INT8 (static QDQ quantization, per-channel weights).

  python scripts/export.py
Outputs models/signnet_fp32.onnx, models/signnet_int8.onnx, models/vocab.json and prints sizes + agreement.
The INT8 model runs on ONNX Runtime (desktop, Android via onnxruntime-android / NNAPI / QNN EP).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echogesture.model import SignNet, example_input  # noqa: E402


class Ensemble(torch.nn.Module):
    """Average member probabilities; output log-probs so the runtime's softmax recovers the mean."""

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def forward(self, x):
        p = torch.stack([torch.softmax(m(x), dim=-1) for m in self.members]).mean(0)
        return torch.log(p + 1e-8)


class Calib(CalibrationDataReader):
    def __init__(self, feats: np.ndarray):
        self.it = iter([{"features": f[None].astype(np.float32)} for f in feats])

    def get_next(self):
        return next(self.it, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", default=[ROOT / "checkpoints" / "model.pt"],
                    help="one or more checkpoints; several are fused into a single averaged-ensemble graph")
    ap.add_argument("--calib", type=Path, default=ROOT / "checkpoints" / "calib.npy")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "models")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    members = []
    for path in args.ckpt:
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        m = SignNet(cfg["num_classes"], dim=cfg["dim"], depth=cfg["depth"])
        m.load_state_dict(ckpt["state_dict"])
        members.append(m.eval())
    model = members[0] if len(members) == 1 else Ensemble(members).eval()
    print(f"exporting {len(members)} model(s)")

    fp32 = args.out_dir / "signnet_fp32.onnx"
    torch.onnx.export(model, (example_input(),), str(fp32), input_names=["features"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(fp32)))

    pre = args.out_dir / "signnet_pre.onnx"
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    int8 = args.out_dir / "signnet_int8.onnx"
    calib = np.load(args.calib)
    quantize_static(str(pre), str(int8), Calib(calib), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    pre.unlink(missing_ok=True)
    (args.out_dir / "vocab.json").write_text(json.dumps(ckpt["vocab"]))

    # agreement check FP32 vs INT8 on the calibration features
    s32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    agree = np.mean([
        s32.run(None, {"features": f[None]})[0].argmax() == s8.run(None, {"features": f[None]})[0].argmax()
        for f in calib.astype(np.float32)
    ])
    mb = lambda p: p.stat().st_size / 1e6  # noqa: E731
    print(f"fp32 {mb(fp32):.2f} MB | int8 {mb(int8):.2f} MB | top-1 agreement fp32 vs int8: {agree:.3f}")


if __name__ == "__main__":
    main()
