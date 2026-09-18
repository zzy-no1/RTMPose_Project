#!/usr/bin/env python3
"""Run preflight checks and create the reproducible Linux server archive."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'rtmpose_server_package.tar.gz'
ARCHIVE_ROOT = 'rtmpose_server_package'

ROOT_FILES = (
    'README.md', 'LICENSE', 'setup.py', 'setup.cfg', 'requirements.txt',
    'SERVER_SETUP.md', 'VERSION_REFERENCE.txt', 'collect_results.py',
    'check_env.sh', 'check_refine_head.sh', 'run_baseline_smoke.sh',
    'run_refine_smoke.sh', 'run_baseline_10e.sh', 'run_refine_10e.sh',
    'build_server_package.py',
)
CONFIG_FILES = (
    'configs/_base_/default_runtime.py',
    'configs/_base_/datasets/coco.py',
    'projects/rtmpose/rtmpose/body_2d_keypoint/'
    'rtmpose-m_smoke_coco-256x192.py',
    'projects/rtmpose/rtmpose/body_2d_keypoint/'
    'rtmpose-m_refine_smoke_coco-256x192.py',
    'projects/rtmpose/rtmpose/body_2d_keypoint/'
    'rtmpose-m_baseline_10e_coco-256x192.py',
    'projects/rtmpose/rtmpose/body_2d_keypoint/'
    'rtmpose-m_refine_10e_coco-256x192.py',
)
TOOL_FILES = ('tools/train.py', 'tools/test.py',
              'tools/check_server_package.py')
ANALYSIS_FILES = (
    'analysis/analyze_error_distribution.py',
    'analysis/analyze_rtmpose_errors.py',
    'analysis/inspect_rtmpose_shapes.py',
    'analysis/visualize_bad_cases.py',
)


def add_file(archive: tarfile.TarFile, relative: str) -> None:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    archive.add(path, arcname=f'{ARCHIVE_ROOT}/{relative}', recursive=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--skip-preflight', action='store_true',
        help='Skip only if tools/check_server_package.py already passed.')
    args = parser.parse_args()
    if not args.skip_preflight:
        subprocess.run(
            [sys.executable, str(ROOT / 'tools/check_server_package.py')],
            cwd=ROOT, check=True)

    with tarfile.open(OUTPUT, 'w:gz', format=tarfile.PAX_FORMAT) as archive:
        for relative in ROOT_FILES + CONFIG_FILES + TOOL_FILES + ANALYSIS_FILES:
            add_file(archive, relative)
        for directory in ('mmpose', 'requirements'):
            base = ROOT / directory
            for path in sorted(base.rglob('*')):
                if not path.is_file():
                    continue
                relative = path.relative_to(ROOT)
                parts = relative.parts
                if ('__pycache__' in parts or '.mim' in parts
                        or path.suffix in {'.pyc', '.pyo'}):
                    continue
                archive.add(
                    path, arcname=f'{ARCHIVE_ROOT}/{relative.as_posix()}',
                    recursive=False)

    print(f'Created {OUTPUT}')


if __name__ == '__main__':
    main()
