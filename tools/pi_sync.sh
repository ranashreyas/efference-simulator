#!/usr/bin/env bash
# Copy the code to the Pi, and optionally a trained model. Run on the Mac.
#
#   tools/pi_sync.sh                # code only
#   tools/pi_sync.sh kitchen_v2     # code + data/models/kitchen_v2/model.onnx -> Pi models/kitchen_v2.onnx
#
# The Mac copy is the master: code files that no longer exist here are removed on the Pi too.
# The Pi's own .venv, recordings/, models/ and events/ (the labeled event store) are never touched.
# Override the Pi with PI=user@host and its project folder (relative to home) with PI_DIR=...
set -euo pipefail

PI="${PI:-ranashreyas@raspberrypi.local}"
PI_DIR="${PI_DIR:-Desktop/efference-simulator}"
cd "$(dirname "$0")/.."

rsync -a --delete \
  --exclude .git --exclude '.venv*' --exclude data --exclude recordings --exclude models --exclude events \
  --exclude __pycache__ --exclude .DS_Store --exclude 'snapshot_*.png' \
  ./ "$PI:$PI_DIR/"
echo "code synced to $PI:$PI_DIR"

if [ $# -ge 1 ]; then
  model="$1"
  onnx="data/models/$model/model.onnx"
  if [ ! -f "$onnx" ]; then
    echo "no $onnx; export it first: .venv-train/bin/python -m train.export --model $model" >&2
    exit 1
  fi
  ssh "$PI" "mkdir -p $PI_DIR/models"
  scp -q "$onnx" "$PI:$PI_DIR/models/$model.onnx"
  echo "model copied to $PI:$PI_DIR/models/$model.onnx"
fi
