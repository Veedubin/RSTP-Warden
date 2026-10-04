#!/usr/bin/env bash
# Fine-tune wildlife-yolox-s on one GPU with YOLOX's own trainer. Run after prepare.py.
#   BATCH=16 ./train.sh            # smaller batch if the GPU runs out of memory
#   WILDLIFE_EPOCHS=80 ./train.sh --resume   # continue a run with more epochs
# Checkpoints land in YOLOX_outputs/wildlife_yolox_s/ (best_ckpt.pth is what export.py uses).
set -euo pipefail
cd "$(dirname "$0")"
[ -f .yolox/yolox/__init__.py ] || { echo "run ./setup.sh first"; exit 1; }
[ -f yolox_s.pth ] || curl -L -o yolox_s.pth \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
export PYTHONPATH="$PWD/.yolox${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --no-sync python .yolox/tools/train.py -f exp.py -d 1 -b "${BATCH:-32}" --fp16 \
  -c yolox_s.pth "$@"
