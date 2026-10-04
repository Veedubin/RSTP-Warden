#!/usr/bin/env bash
# One-time setup: the training venv plus the pinned YOLOX source checkout (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
YOLOX_COMMIT=6ddff4824372906469a7fae2dc3206c7aa4bbaee
uv sync
if [ ! -f .yolox/yolox/__init__.py ]; then
  rm -rf .yolox
  git init -q .yolox
  git -C .yolox remote add origin https://github.com/Megvii-BaseDetection/YOLOX.git
  git -C .yolox fetch -q --depth 1 origin "$YOLOX_COMMIT"
  git -C .yolox checkout -q FETCH_HEAD
fi
echo "yolox checkout: $(git -C .yolox rev-parse HEAD)"
uv run --no-sync python -c "import sys; sys.path.insert(0, '.yolox'); import torch, yolox; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| yolox', yolox.__version__)"
