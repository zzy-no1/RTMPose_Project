# RTMPose-M Linux GPU 服务器复现实验包

本实验包包含两组严格配对实验：原始 `RTMCCHead` baseline，以及在
`final_layer` 后、flatten 前加入 depthwise 3×3 Conv + BatchNorm + SiLU +
residual 的 `RTMCCRefineHead` V1。10 epoch 配置除 Head 类型外保持一致。

请在 Linux 服务器上按下述步骤重新创建环境；不要复制或直接迁移 Windows
Conda 环境，因为操作系统、CUDA 扩展和二进制依赖并不兼容。

## 1. 解压与创建环境

```bash
tar -xzf rtmpose_server_package.tar.gz
cd rtmpose_server_package

conda create -n rtmpose-server python=3.10.20 -y
conda activate rtmpose-server
python -m pip install --upgrade pip setuptools wheel
```

安装 CUDA 12.8 版 PyTorch：

```bash
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
```

安装已验证的核心版本和 RTMPose 所需依赖：

```bash
pip install numpy==1.23.5 mmengine==0.10.7
pip install mmcv==2.1.0
pip install mmdet==3.2.0
pip install albumentations==1.3.1 opencv-python==4.8.1.78
pip install xtcocotools scipy json_tricks munkres matplotlib chumpy
pip install -e . --no-deps
```

`mmcv==2.1.0` 若没有适配当前 PyTorch/CUDA 的预编译 wheel，会在服务器上
编译。此时需预先安装与服务器 CUDA 匹配的 toolkit、gcc/g++、ninja；也可使用
服务器管理员已经验证的 mmcv 2.1.0 wheel。版本基准见
`VERSION_REFERENCE.txt`。

## 2. 准备 COCO 数据

实验配置固定从仓库根目录的 `data/coco/` 读取数据。数据不在压缩包内，请复制
或建立软链接，使目录至少包含：

```text
data/coco/
├── annotations/
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── person_detection_results/
│   └── COCO_val2017_detections_AP_H_56_person.json
├── train2017/
└── val2017/
```

例如：

```bash
mkdir -p data
ln -s /path/to/COCO data/coco
```

## 3. 上传后的第一步：环境与代码自检

```bash
chmod +x ./*.sh
./check_env.sh
./check_refine_head.sh
```

第二个命令会自动检查：Head 注册、四份配置加载、baseline/refine 配对一致性、
refine 模型 build，以及 `[B,17,8,6] -> refine -> [B,17,8,6]`。

## 4. 启动实验

服务器首次运行的推荐顺序固定为：

```text
check_env.sh → check_refine_head.sh → baseline smoke → refine smoke → baseline 10e → refine 10e → collect_results.py
```

对应命令如下，应在解压后的项目根目录依次执行：

```bash
./check_env.sh
./check_refine_head.sh
./run_baseline_smoke.sh
./run_refine_smoke.sh
./run_baseline_10e.sh
./run_refine_10e.sh
python collect_results.py
```

默认单卡、batch size 8、workers 4：

```bash
./run_baseline_smoke.sh
./run_refine_smoke.sh
./run_baseline_10e.sh
./run_refine_10e.sh
```

可在命令前覆盖 GPU、batch size 和 workers。例如 batch size 16：

```bash
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=16 NUM_WORKERS=8 ./run_baseline_10e.sh
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=16 NUM_WORKERS=8 ./run_refine_10e.sh
```

batch size 32 时将 `BATCH_SIZE=32`。当 `NUM_WORKERS=0` 时，脚本会自动将
`persistent_workers` 设为 false；workers 大于 0 时设为 true。四个实验分别写入：

- `work_dirs/server_baseline_smoke`
- `work_dirs/server_refine_smoke`
- `work_dirs/server_baseline_10e`
- `work_dirs/server_refine_10e`

首次运行需要下载相同的 CSPNeXt-M 预训练 backbone。离线服务器应提前把权重放入
PyTorch cache，或在两份 10 epoch 配置中将同一个 `checkpoint` URL 同步改成本地
路径。

## 5. 汇总 epoch 5/10 指标

两组 10 epoch 实验结束后运行：

```bash
python collect_results.py
```

默认生成 `rtmpose_10e_comparison.csv`，包含 `coco/AP`、AP50、AP75、APM、
APL 和 AR。也可指定目录与输出文件：

```bash
python collect_results.py \
  --baseline-dir work_dirs/server_baseline_10e \
  --refine-dir work_dirs/server_refine_10e \
  --output comparison.csv
```

## 6. 误差分析

`analysis/` 中保留了现有四个误差分析脚本，供后续比较 hip/knee/ankle/wrist、
medium vs large、visible vs occluded，以及 P90/P95/P99。大型中间结果、图片输出和
pickle 文件未收入迁移包；服务器上应使用新的实验输出重新生成。

## 7. 包内容与排除项

迁移包只收入训练所需的 `mmpose` Python 源码、四份配置、COCO metadata、训练/
测试入口、误差分析脚本、运行脚本及文档。明确不包含 `data/`、`work_dirs/`、
`.git/`、`__pycache__/`、checkpoint、`.pth`、`.pkl` 和既有分析结果图片。
