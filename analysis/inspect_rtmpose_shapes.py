from mmengine.config import Config

from mmpose.registry import MODELS
from mmpose.utils import register_all_modules

import torch


# ============================================================
# Config
# ============================================================

CONFIG = (
    r'projects\rtmpose\rtmpose\body_2d_keypoint'
    r'\rtmpose-m_8xb256-420e_coco-256x192.py'
)


# ============================================================
# 1. 注册 MMPose 模块
# ============================================================

register_all_modules(
    init_default_scope=True
)


# ============================================================
# 2. 加载配置
# ============================================================

cfg = Config.fromfile(CONFIG)


# ============================================================
# 3. 构建模型
# ============================================================

model = MODELS.build(
    cfg.model
)

model.eval()
model.cuda()


# ============================================================
# Hook
# ============================================================

def make_hook(name):

    def hook(module, inputs, output):

        print()
        print("=" * 60)
        print(name)
        print("=" * 60)

        print("INPUT:")

        for i, x in enumerate(inputs):

            if isinstance(x, torch.Tensor):

                print(
                    f"  input[{i}]:",
                    tuple(x.shape)
                )

            elif isinstance(x, (list, tuple)):

                for j, item in enumerate(x):

                    if isinstance(
                        item,
                        torch.Tensor
                    ):

                        print(
                            f"  input[{i}][{j}]:",
                            tuple(item.shape)
                        )

        print("OUTPUT:")

        if isinstance(output, torch.Tensor):

            print(
                " ",
                tuple(output.shape)
            )

        elif isinstance(output, (list, tuple)):

            for i, item in enumerate(output):

                if isinstance(
                    item,
                    torch.Tensor
                ):

                    print(
                        f"  output[{i}]:",
                        tuple(item.shape)
                    )

    return hook


# ============================================================
# 注册各模块 Hook
# ============================================================

model.backbone.register_forward_hook(
    make_hook(
        "1. CSPNeXt BACKBONE"
    )
)

model.head.final_layer.register_forward_hook(
    make_hook(
        "2. RTMCCHead FINAL_LAYER"
    )
)

model.head.mlp.register_forward_hook(
    make_hook(
        "3. RTMCCHead MLP"
    )
)

model.head.gau.register_forward_hook(
    make_hook(
        "4. GAU / RTMCCBlock"
    )
)

model.head.cls_x.register_forward_hook(
    make_hook(
        "5. SimCC CLS_X"
    )
)

model.head.cls_y.register_forward_hook(
    make_hook(
        "6. SimCC CLS_Y"
    )
)


# ============================================================
# 构造一张假的人体输入
#
# BCHW:
# B = 1
# C = 3
# H = 256
# W = 192
# ============================================================

x = torch.randn(
    1,
    3,
    256,
    192,
    device="cuda"
)


print()
print("=" * 60)
print("RTMPOSE-M SHAPE INSPECTION")
print("=" * 60)

print(
    "Input:",
    tuple(x.shape)
)


# ============================================================
# Forward
# ============================================================

with torch.no_grad():

    feats = model.backbone(
        x
    )

    pred_x, pred_y = (
        model.head.forward(
            feats
        )
    )


# ============================================================
# 最终输出
# ============================================================

print()
print("=" * 60)
print("FINAL OUTPUT")
print("=" * 60)

print(
    "pred_x:",
    tuple(pred_x.shape)
)

print(
    "pred_y:",
    tuple(pred_y.shape)
)

print()
print("DONE")