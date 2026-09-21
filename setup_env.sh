#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-rtmpose-cu118}"
source /root/miniconda3/etc/profile.d/conda.sh

conda create -n "${ENV_NAME}" python=3.10.20 -y
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip wheel
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.23.5
pip install mmengine==0.10.7
pip install openmim==0.3.9
mim install "mmcv==2.1.0"

pip uninstall -y opencv-python opencv-python-headless || true
pip install numpy==1.23.5
pip install opencv-python==4.8.1.78 --no-deps

pip install mmdet==3.2.0
pip install albumentations==1.3.1
pip install xtcocotools==1.14.3
pip install json_tricks==3.17.3
pip install munkres==1.1.4
pip install cython==3.3.0
pip install chumpy==0.70 --no-build-isolation

pip install setuptools==75.8.0
pip install -e . --no-deps --no-build-isolation

python -c "import torch,numpy,cv2,mmcv,mmengine,mmdet,mmpose,albumentations; print('torch:',torch.__version__); print('cuda:',torch.version.cuda); print('numpy:',numpy.__version__); print('opencv:',cv2.__version__); print('mmcv:',mmcv.__version__); print('mmengine:',mmengine.__version__); print('mmdet:',mmdet.__version__); print('mmpose:',mmpose.__version__); print('albumentations:',albumentations.__version__)"
