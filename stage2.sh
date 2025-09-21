#!/usr/bin/env bash

set -euo pipefail

DATASET_ROOT=${1:-/path/to/DATASET_ROOT}
STAGE1_MODEL_DIR=${2:-./output/<dataset_name>/<stage1_timestamp>}
DEPTHS_DIR=${3:-images_depth}

# RGB+Thermal finetuning with smoke decomposition
python train_finetune_thermal.py --eval -s "$DATASET_ROOT" -m "$STAGE1_MODEL_DIR" --use_thermal --depths "$DEPTHS_DIR"
