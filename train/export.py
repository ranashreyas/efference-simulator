"""Export a trained model to ONNX for the Pi. Run on the Mac.

    .venv-train/bin/python -m train.export --model kitchen_v2

Writes data/models/<model>/model.onnx and onnx_check.npz (a test input plus PyTorch's outputs,
so the Pi can confirm ONNX Runtime gives the same answer).
ONNX inputs:  frames (1, 3, 96, 128) float32 in [0, 1], motor (1, 35) float32 (eye/motor.py)
ONNX outputs: pred (1, 1, 96, 128), logvar (1, 1, 96, 128)
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from train.model import ACTION_DIM, N_FRAMES, load_model

MODELS = Path(__file__).resolve().parent.parent / "data" / "models"


class Exported(torch.nn.Module):
    """Only the outputs the live loop needs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frames, motor):
        pred, logvar, _ = self.model(frames, motor)
        return pred, logvar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    mdir = MODELS / args.model
    model = load_model(mdir / "model.pt")
    wrapped = Exported(model).eval()

    frames = torch.rand(1, N_FRAMES, 96, 128)
    motor = torch.randn(1, ACTION_DIM) * 0.3
    torch.onnx.export(wrapped, (frames, motor), str(mdir / "model.onnx"), opset_version=17,
                      input_names=["frames", "motor"], output_names=["pred", "logvar"])
    with torch.no_grad():
        pred, logvar = wrapped(frames, motor)
    np.savez(mdir / "onnx_check.npz", frames=frames.numpy(), motor=motor.numpy(),
             pred=pred.numpy(), logvar=logvar.numpy())
    print(f"wrote {mdir / 'model.onnx'} ({(mdir / 'model.onnx').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
