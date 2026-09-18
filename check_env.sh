#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

echo "== System =="
uname -a
command -v nvidia-smi >/dev/null && nvidia-smi || true

echo "== Python packages =="
"${PYTHON}" - <<'PY'
import importlib
import platform
import sys

packages = (
    'torch', 'numpy', 'mmcv', 'mmengine', 'mmpose', 'mmdet',
    'albumentations', 'cv2')
print('python:', sys.version.replace('\n', ' '))
print('platform:', platform.platform())
for name in packages:
    try:
        module = importlib.import_module(name)
        print(f'{name}: {getattr(module, "__version__", "unknown")}')
    except Exception as exc:
        print(f'{name}: ERROR: {exc}')

import torch
print('cuda_available:', torch.cuda.is_available())
print('torch_cuda:', torch.version.cuda)
print('cudnn:', torch.backends.cudnn.version())
print('gpu_count:', torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(f'gpu_{index}: {torch.cuda.get_device_name(index)}')
if not torch.cuda.is_available():
    raise SystemExit('ERROR: PyTorch cannot access a CUDA GPU.')
PY

