# RTMPose_Project 环境说明

## 已验证核心环境

```text
Ubuntu 22.04
Python 3.10.20
PyTorch 2.1.2+cu118
TorchVision 0.16.2+cu118
MMCV 2.1.0
MMEngine 0.10.7
MMDetection 3.2.0
MMPose 1.3.2
NumPy 1.23.5
OpenCV 4.8.1
Albumentations 1.3.1
chumpy 0.70
xtcocotools 1.14.3
json_tricks 3.17.3
munkres 1.1.4
Cython 3.3.0
```

## 安装步骤

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n rtmpose-cu118 python=3.10.20 -y
conda activate rtmpose-cu118
```

PyTorch：

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

基础依赖：

```bash
pip install numpy==1.23.5
pip install mmengine==0.10.7
pip install openmim==0.3.9
mim install "mmcv==2.1.0"
```

重新固定 NumPy / OpenCV：

```bash
pip uninstall -y opencv-python opencv-python-headless
pip install numpy==1.23.5
pip install opencv-python==4.8.1.78 --no-deps
```

安装其余依赖：

```bash
pip install mmdet==3.2.0
pip install albumentations==1.3.1
pip install xtcocotools==1.14.3
pip install json_tricks==3.17.3
pip install munkres==1.1.4
pip install cython==3.3.0
pip install chumpy==0.70 --no-build-isolation
```

安装本仓库：

```bash
pip install setuptools==75.8.0
pip install -e . --no-deps --no-build-isolation
```

综合检查：

```bash
python -c "import torch,numpy,cv2,mmcv,mmengine,mmdet,mmpose,albumentations; print('torch:',torch.__version__); print('cuda:',torch.version.cuda); print('numpy:',numpy.__version__); print('opencv:',cv2.__version__); print('mmcv:',mmcv.__version__); print('mmengine:',mmengine.__version__); print('mmdet:',mmdet.__version__); print('mmpose:',mmpose.__version__); print('albumentations:',albumentations.__version__)"
```

V1 注册检查：

```bash
python -c "from mmpose.utils import register_all_modules; register_all_modules(); from mmpose.registry import MODELS; print(MODELS.get('RTMCCRefineHead'))"
```
