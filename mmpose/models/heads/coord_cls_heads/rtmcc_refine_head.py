import torch
import torch.nn as nn

from mmpose.registry import MODELS
from .rtmcc_head import RTMCCHead


@MODELS.register_module()
class RTMCCRefineHead(RTMCCHead):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        k = self.out_channels

        self.refine = nn.Sequential(
            nn.Conv2d(
                k,
                k,
                kernel_size=3,
                padding=1,
                groups=k,
                bias=False
            ),
            nn.BatchNorm2d(k),
            nn.SiLU(inplace=True)
        )

    def forward(self, feats):
        feats = feats[-1]

        feats = self.final_layer(feats)

        refined = self.refine(feats)
        feats = feats + refined

        feats = torch.flatten(feats, 2)
        feats = self.mlp(feats)
        feats = self.gau(feats)

        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)

        return pred_x, pred_y
