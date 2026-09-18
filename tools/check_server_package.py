#!/usr/bin/env python3
"""Preflight checks for the RTMPose baseline/refine server experiments."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

import mmpose.models  # noqa: F401 -- register model components
from mmpose.registry import MODELS


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'projects/rtmpose/rtmpose/body_2d_keypoint'
CONFIGS = {
    'baseline_smoke': CONFIG_DIR / 'rtmpose-m_smoke_coco-256x192.py',
    'refine_smoke': CONFIG_DIR / 'rtmpose-m_refine_smoke_coco-256x192.py',
    'baseline_10e': CONFIG_DIR / 'rtmpose-m_baseline_10e_coco-256x192.py',
    'refine_10e': CONFIG_DIR / 'rtmpose-m_refine_10e_coco-256x192.py',
}


def comparable(cfg: Config) -> dict:
    value = copy.deepcopy(cfg.to_dict())
    value['model']['head'].pop('type')
    return value


def main() -> None:
    init_default_scope('mmpose')
    refine_class = MODELS.get('RTMCCRefineHead')
    assert refine_class is not None, 'RTMCCRefineHead is not registered'
    print('[PASS] RTMCCRefineHead is registered')

    configs = {}
    for name, path in CONFIGS.items():
        configs[name] = Config.fromfile(path)
        print(f'[PASS] config readable: {name}')

    for suffix in ('smoke', '10e'):
        baseline = configs[f'baseline_{suffix}']
        refine = configs[f'refine_{suffix}']
        assert baseline.model.head.type == 'RTMCCHead'
        assert refine.model.head.type == 'RTMCCRefineHead'
        assert comparable(baseline) == comparable(refine), (
            f'{suffix} configs differ outside model.head.type')
        print(f'[PASS] {suffix} pair differs only by head type')

    for name in ('baseline_10e', 'refine_10e'):
        cfg = configs[name]
        assert cfg.train_cfg.max_epochs == 10
        assert cfg.train_cfg.val_interval == 5
        assert cfg.train_dataloader.batch_size == 8
        assert cfg.train_dataloader.num_workers == 4
        assert cfg.train_dataloader.persistent_workers is True
        assert cfg.randomness.seed == 2026
    print('[PASS] 10-epoch runtime settings are reproducible and matched')

    model = MODELS.build(configs['refine_10e'].model)
    assert model.head.__class__.__name__ == 'RTMCCRefineHead'
    print('[PASS] refine model builds')

    feature_map = torch.randn(2, 17, 8, 6)
    model.head.refine.eval()
    with torch.no_grad():
        refined = feature_map + model.head.refine(feature_map)
    assert refined.shape == feature_map.shape
    print('[PASS] shape: [2, 17, 8, 6] -> refine -> [2, 17, 8, 6]')
    print('All server-package preflight checks passed.')


if __name__ == '__main__':
    main()
