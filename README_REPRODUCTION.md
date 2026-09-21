# RTMPose_Project

本仓库用于硕士论文阶段的 RTMPose-M 人体姿态估计研究，当前主要包含原始 `RTMCCHead` Baseline、轻量局部空间细化模块 `RTMCCRefineHead`（V1）、严格配对实验配置、Linux/AutoDL 运行脚本、环境检查与误差分析脚本。

## 1. 克隆项目

```bash
git clone https://github.com/zzy-no1/RTMPose_Project.git
cd RTMPose_Project
```

## 2. 已验证环境

- Ubuntu 22.04
- Python 3.10.20
- PyTorch 2.1.2+cu118
- TorchVision 0.16.2+cu118
- CUDA runtime 11.8
- MMCV 2.1.0
- MMEngine 0.10.7
- MMDetection 3.2.0
- MMPose 1.3.2（本仓库 editable install）
- NumPy 1.23.5
- OpenCV 4.8.1
- Albumentations 1.3.1
- RTX 4090 24GB（已验证）

详细安装见 `ENVIRONMENT.md`，也可参考 `setup_env.sh`。

## 3. 数据目录

```text
RTMPose_Project/
└── data/
    └── coco/
        ├── annotations/
        │   ├── person_keypoints_train2017.json
        │   └── person_keypoints_val2017.json
        ├── person_detection_results/
        │   └── COCO_val2017_detections_AP_H_56_person.json
        ├── train2017/
        └── val2017/
```

COCO 数据不包含在 GitHub 仓库中。

如果实际数据位于：

```text
/root/autodl-tmp/datasets/coco
```

可创建软链接：

```bash
mkdir -p data
ln -s /root/autodl-tmp/datasets/coco data/coco
```

检查：

```bash
find data/coco/train2017 -maxdepth 1 -type f | wc -l
find data/coco/val2017 -maxdepth 1 -type f | wc -l
```

正常数量：

```text
train2017: 118287
val2017: 5000
```

## 4. 环境和代码自检

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtmpose-cu118
chmod +x ./*.sh
./check_env.sh
./check_refine_head.sh
```

## 5. Smoke Test

```bash
./run_baseline_smoke.sh
./run_refine_smoke.sh
```

## 6. 10 Epoch 实验

```bash
./run_baseline_10e.sh
./run_refine_10e.sh
```

已完成的一组严格配对结果：

| Metric | Baseline 10e | V1 10e |
|---|---:|---:|
| AP | 0.321572 | 0.328712 |
| AP50 | 0.667758 | 0.668024 |
| AP75 | 0.268082 | 0.282417 |
| APM | 0.332421 | 0.338891 |
| APL | 0.325471 | 0.334027 |
| AR | 0.390538 | 0.398253 |

## 7. 50 Epoch / Batch 64 实验

配置：

```text
projects/rtmpose/rtmpose/body_2d_keypoint/
├── rtmpose-m_baseline_50e_bs64_coco-256x192.py
└── rtmpose-m_refine_50e_bs64_coco-256x192.py
```

运行：

```bash
./run_baseline_50e_bs64.sh
./run_refine_50e_bs64.sh
```

当前 50e 关键参数：

```text
max_epochs = 50
stage2_num_epochs = 4
batch_size = 64
num_workers = 4
base_lr = 4e-3
val_interval = 5
seed = 2026

LinearLR: begin=0, end=125
CosineAnnealingLR: begin=25, end=50, T_max=25, eta_min=2e-4
```

Baseline 与 V1 除 Head 类型外应保持严格一致。

## 8. 长训练建议使用 screen

```bash
screen -S baseline50e
```

进入后：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtmpose-cu118
cd /root/autodl-tmp/RTMPose_Project
./run_baseline_50e_bs64.sh
```

Detach：`Ctrl+A`，松开，再按 `D`。

重新进入：

```bash
screen -r baseline50e
```

## 9. Batch Benchmark

| Batch | Workers | time/iter | 吞吐量约 |
|---:|---:|---:|---:|
| 8 | 4 | ~0.095 s | ~84 samples/s |
| 16 | 4 | ~0.113 s | ~142 samples/s |
| 32 | 4 | ~0.181 s | ~177 samples/s |
| 64 | 4 | ~0.270 s | ~237 samples/s |
| 64 | 8 | ~0.290 s | ~221 samples/s |
| 64 | 2 | ~0.288–0.296 s | ~216–222 samples/s |
| 128 | 4 | ~0.566 s | ~226 samples/s |

当前吞吐甜点约为 `batch=64, workers=4`。

## 10. 结果与误差分析

结果目录：

```text
work_dirs/server_baseline_10e
work_dirs/server_refine_10e
work_dirs/server_baseline_50e_bs64
work_dirs/server_refine_50e_bs64
```

已有分析脚本：

```text
analysis/analyze_rtmpose_errors.py
analysis/analyze_error_distribution.py
analysis/inspect_rtmpose_shapes.py
analysis/visualize_bad_cases.py
```

后续重点比较 wrist、ankle、knee、hip，以及 visible/occluded、medium/large、P50/P90/P95/P99。

## 11. GitHub 默认不包含

```text
COCO 数据
person_detection_results
work_dirs
*.pth
*.pkl
大型日志和可视化中间结果
```

如需直接测试已有训练权重，请单独获取 checkpoint。

## 12. V1 模块

```text
final_layer output [B,17,8,6]
→ Depthwise 3×3 Conv
→ BatchNorm
→ SiLU
→ Residual
→ RTMCC coordinate classification
```

## 13. 推荐实验顺序

```text
环境检查
→ smoke
→ 10e
→ 50e
→ 关键点级误差分析
→ 消融实验
→ 100e / 更长周期验证
```
