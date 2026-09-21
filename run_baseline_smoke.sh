#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PYTHON="${PYTHON:-python}"
CONFIG="projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_smoke_coco-256x192.py"
WORK_DIR="work_dirs/server_baseline_smoke"
PERSISTENT_WORKERS=true
if [[ "${NUM_WORKERS}" -eq 0 ]]; then PERSISTENT_WORKERS=false; fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
"${PYTHON}" tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" --cfg-options \
  train_dataloader.batch_size="${BATCH_SIZE}" \
  train_dataloader.num_workers="${NUM_WORKERS}" \
  train_dataloader.persistent_workers="${PERSISTENT_WORKERS}" \
  val_dataloader.num_workers="${NUM_WORKERS}" \
  val_dataloader.persistent_workers="${PERSISTENT_WORKERS}" \
  test_dataloader.num_workers="${NUM_WORKERS}" \
  test_dataloader.persistent_workers="${PERSISTENT_WORKERS}"

