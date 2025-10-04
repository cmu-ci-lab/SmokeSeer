## SmokeSeer: 3D Gaussian Splatting for Joint Smoke Removal and Scene Reconstruction

### Overview
SmokeSeer jointly reconstructs a smoke-filled scene and removes smoke using 3D Gaussian Splatting, leveraging RGB and thermal images. Thermal provides robust geometry cues through smoke; RGB provides texture. We decompose the scene into surface and smoke Gaussians and render smoke-free views by omitting smoke Gaussians.

### Resources
- [Project website](https://imaging.cs.cmu.edu/smokeseer/)
- [Dataset folder](https://drive.google.com/drive/folders/1xwmLNRIaZYPERdSBJVWpu5P80DSEu6Ts?usp=sharing)

real.zip contains the colmap parameters and the frames used for real-world experiments in the paper. We also include the raw videos from which we extracted the frames. 

synthetic.zip contains the blender camera parameters and the frames used for synthetic data experiments in the paper. 

### Quick Start
1. Environment
- Install Conda (Miniconda/Anaconda).
- Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate gaussian_splatting
```
- Install extras (if not already present):
```bash
bash INSTALL.sh
```

2. Data Preparation (COLMAP-style)
Your dataset directory should look like:
```text
DATASET_ROOT/
  images/  (RGB and thermal images; thermal filenames should include the word "thermal")
  sparse/0/
    cameras.bin|txt, images.bin|txt, points3D.bin|txt
  images_mask/   (optional, smoke masks per RGB image: same filenames as in images/, .png)
  images_depth/  (optional, inverse-depth PNGs from monocular depth)
```

3. (Optional) Generate thermal monocular depth and scale it
- Place the thermal images under `images/` with names containing "thermal".
- Download or place the required model weights under `checkpoints/` as needed by Depth-Anything-V2.
- Run:
```bash
bash get_depth.sh DATASET_ROOT
```
This writes inverse-depth PNGs to `DATASET_ROOT/images_depth` and a scale file to `DATASET_ROOT/sparse/0/depth_params.json`.

### Training
- Stage 1 (Thermal-only surface reconstruction)  
  Learns the scene geometry from thermal views and saves outputs under `./output/<dataset_name>/<timestamp>`.
  - Example:
```bash
python train_stage1_thermal.py -s DATASET_ROOT --eval --use_thermal --depths images_depth
```

- Stage 2 (RGB + Thermal finetuning with smoke decomposition)  
  Optimizes both surface and smoke Gaussians. Pass the Stage 1 model path (`-m`) you want to finetune from.
  - Example:
```bash
python train_finetune_thermal.py --eval -s DATASET_ROOT -m ./output/<dataset_name>/<stage1_timestamp> --use_thermal --depths images_depth
```

### Convenience scripts
```bash
# Stage 1
bash stage1.sh DATASET_ROOT

# Stage 2
bash stage2.sh DATASET_ROOT ./output/<dataset_name>/<stage1_timestamp>

# Depth
bash get_depth.sh DATASET_ROOT
```

### Reproducing our runs
We primarily used commands equivalent to:
```bash
# Stage 1 (thermal-only)
python train_stage1_thermal.py -s DATASET_ROOT --eval --use_thermal --depths images_depth

# Stage 2 (finetune with RGB+thermal)
python train_finetune_thermal.py --eval -s DATASET_ROOT -m ./output/<dataset_name>/<stage1_timestamp> --use_thermal --depths images_depth
```
Flags like `smoke_opacity_weight`, `smoke_color_weight`, `dcp_weight`, `densify_grad_threshold_surface`, `densify_grad_threshold_smoke`, and `smoke_uniformity_color_weight` can be tuned per-scene (see the script defaults in `arguments/` and `train_finetune_thermal.py`). We disable W&B by default; enable with `--use_wandb` if desired.

### Expected filenames
- Thermal images: filenames containing "thermal" (e.g., `frame_0001_thermal.png`). Non-thermal images are treated as RGB.
- Optional per-image smoke mask for RGB: `images_mask/<same_name>.png`
- Depth PNGs: `images_depth/<same_name>.png` (16-bit inverse depth)

### Outputs
- `./output/<dataset_name>/<timestamp>/`
  - `cfg_args` (run config)
  - `cameras.json`
  - `checkpoints/`
  - `point_cloud/`
  - test results (if enabled)

Stage 2 additionally saves smoke/thermal-specific checkpoints and optional test renders.

### Notes
- This repo includes third-party components (see `LICENSE.md`). Our usage is for non-commercial research.
- By default, large folders like `output/` and `wandb/` are git-ignored. Scripts are sanitized to avoid user-specific absolute paths.

### Citation
If you use this repository, please cite our SmokeSeer paper:

Paper: [arXiv:2509.17329](https://arxiv.org/abs/2509.17329)

```bibtex
@article{jain2025smokeseer,
  title={{SmokeSeer: 3D Gaussian Splatting for Smoke Removal and Scene Reconstruction}},
  author={Neham Jain and Andrew Jong and Sebastian Scherer and Ioannis Gkioulekas},
  year={2025},
  eprint={2509.17329},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2509.17329},
}
```

