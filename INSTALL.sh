#!/usr/bin/env bash

set -euo pipefail

# Install local submodules and common Python deps
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn

pip install av plyfile imageio imageio-ffmpeg opencv-python scikit-learn
pip install git+https://github.com/nerfstudio-project/gsplat.git

# Optional: Depth-Anything-V2 requirements (if not vendored)
if [ -f Depth-Anything-V2/requirements.txt ]; then
  pip install -r Depth-Anything-V2/requirements.txt || true
fi