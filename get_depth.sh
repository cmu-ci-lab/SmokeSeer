#!/usr/bin/env bash

set -euo pipefail

DATASET_ROOT=${1:-/path/to/DATASET_ROOT}

# 1) Predict inverse depth for thermal images
python Depth-Anything-V2/run.py \
  --encoder vitl \
  --pred-only \
  --grayscale \
  --img-path "$DATASET_ROOT/images" \
  --outdir "$DATASET_ROOT/images_depth"

# 2) Compute per-image scale/offset using COLMAP sparse model
python utils/make_depth_scale.py \
  --base_dir "$DATASET_ROOT" \
  --depths_dir "$DATASET_ROOT/images_depth"