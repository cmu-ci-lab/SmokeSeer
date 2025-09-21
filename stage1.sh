#!/usr/bin/env bash

set -euo pipefail

DATASET_ROOT=${1:-/path/to/DATASET_ROOT}
DEPTHS_DIR=${2:-images_depth}

# Thermal-only Stage 1 surface reconstruction
python train_stage1_thermal.py -s "$DATASET_ROOT" --eval --use_thermal --depths "$DEPTHS_DIR"
python train_finetune_thermal.py --eval -s /home/neham/wildfire_all_data/real/drone_AFCA/align_rgb_thermal/red_container_colmap/undistorted -m ./output/undistorted/20250217-23-16-40 --smoke_opacity_weight=0.986319292283202 --smoke_color_weight=0.9627407564287004 --dcp_weight=0.003089720479699798 --densify_grad_threshold_surface=0.00034882024789871546 --densify_grad_threshold_smoke=0.00037892492544383625 --smoke_uniformity_color_weight=3.1030414947610216 --thermal_weight=2.2569701466273075 --use_thermal --use_wandb --depths /home/neham/wildfire_all_data/real/drone_AFCA/align_rgb_thermal/red_container_colmap/undistorted/images_depth