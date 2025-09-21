import random


run_path = "nehamjain/gaussian_desmoking/apt1d1f8"
null = None
false = False
true = True
args = {
  "ip": {
    "desc": null,
    "value": "127.0.0.1"
  },
  "tag": {
    "desc": null,
    "value": "example"
  },
  "eval": {
    "desc": null,
    "value": true
  },
  "port": {
    "desc": null,
    "value": 6009
  },
  "debug": {
    "desc": null,
    "value": false
  },
  "quiet": {
    "desc": null,
    "value": false
  },
  "_wandb": {
    "desc": null,
    "value": {
      "t": {
        "1": [
          1,
          5,
          41,
          53,
          55
        ],
        "2": [
          1,
          5,
          41,
          53,
          55
        ],
        "3": [
          13,
          14,
          16,
          23,
          37,
          61
        ],
        "4": "3.10.16",
        "5": "0.19.4",
        "8": [
          5
        ],
        "13": "linux-x86_64"
      },
      "framework": "torch",
      "start_time": 1740935877,
      "cli_version": "0.19.4",
      "is_jupyter_run": false,
      "python_version": "3.10.16",
      "is_kaggle_kernel": false
    }
  },
  "depths": {
    "desc": null,
    "value": "/home/neham/wildfire_all_data/real/drone_AFCA/align_rgb_thermal/red_container_colmap/undistorted/images_depth"
  },
  "images": {
    "desc": null,
    "value": "images"
  },
  "is_6dof": {
    "desc": null,
    "value": false
  },
  "warm_up": {
    "desc": null,
    "value": 3000
  },
  "pose_opt": {
    "desc": null,
    "value": false
  },
  "tv_weight": {
    "desc": null,
    "value": 0.00001
  },
  "use_wandb": {
    "desc": null,
    "value": true
  },
  "dcp_weight": {
    "desc": null,
    "value": 0.0005
  },
  "debug_from": {
    "desc": null,
    "value": -1
  },
  "feature_lr": {
    "desc": null,
    "value": 0.0025
  },
  "is_blender": {
    "desc": null,
    "value": false
  },
  "model_path": {
    "desc": null,
    "value": "./output/undistorted/20250217-23-16-40"
  },
  "opacity_lr": {
    "desc": null,
    "value": 0.025
  },
  "patch_size": {
    "desc": null,
    "value": 16
  },
  "resolution": {
    "desc": null,
    "value": -1
  },
  "scaling_lr": {
    "desc": null,
    "value": 0.005
  },
  "data_device": {
    "desc": null,
    "value": "cuda"
  },
  "rotation_lr": {
    "desc": null,
    "value": 0.001
  },
  "source_path": {
    "desc": null,
    "value": "/home/neham/wildfire_all_data/real/drone_AFCA/align_rgb_thermal/red_container_colmap/undistorted"
  },
  "use_thermal": {
    "desc": null,
    "value": true
  },
  "disable_save": {
    "desc": null,
    "value": false
  },
  "lambda_dssim": {
    "desc": null,
    "value": 0.2
  },
  "percent_dense": {
    "desc": null,
    "value": 0.01
  },
  "detect_anomaly": {
    "desc": null,
    "value": false
  },
  "opacity_weight": {
    "desc": null,
    "value": 0.05
  },
  "scaling_weight": {
    "desc": null,
    "value": 0.1
  },
  "thermal_weight": {
    "desc": null,
    "value": 3.9217289119100744
  },
  "save_iterations": {
    "desc": null,
    "value": [
      7000,
      15000,
      22000,
      30000
    ]
  },
  "test_iterations": {
    "desc": null,
    "value": [
      1000,
      2000,
      5000,
      7000,
      12000,
      15000,
      23000,
      30000
    ]
  },
  "position_lr_init": {
    "desc": null,
    "value": 0.00016
  },
  "sh_degree_stage1": {
    "desc": null,
    "value": 0
  },
  "sh_degree_stage2": {
    "desc": null,
    "value": 0
  },
  "start_checkpoint": {
    "desc": null,
    "value": "./output/undistorted/20250217-23-16-40/chkpnt_surface15000.pth"
  },
  "white_background": {
    "desc": null,
    "value": false
  },
  "densify_from_iter": {
    "desc": null,
    "value": 500
  },
  "iterations_stage1": {
    "desc": null,
    "value": 15000
  },
  "iterations_stage2": {
    "desc": null,
    "value": 30000
  },
  "position_lr_final": {
    "desc": null,
    "value": 0.0000016
  },
  "random_background": {
    "desc": null,
    "value": false
  },
  "convert_SHs_python": {
    "desc": null,
    "value": false
  },
  "deformation_weight": {
    "desc": null,
    "value": 0.1
  },
  "densify_until_iter": {
    "desc": null,
    "value": 22500
  },
  "max_opacity_weight": {
    "desc": null,
    "value": 0.1
  },
  "min_opacity_weight": {
    "desc": null,
    "value": 0.1
  },
  "smoke_color_weight": {
    "desc": null,
    "value": 1.0444821124208143
  },
  "deform_lr_max_steps": {
    "desc": null,
    "value": 20000
  },
  "load2gpu_on_the_fly": {
    "desc": null,
    "value": false
  },
  "valid_region_weight": {
    "desc": null,
    "value": 0.34828620860972404
  },
  "compute_cov3D_python": {
    "desc": null,
    "value": false
  },
  "depth_l1_weight_init": {
    "desc": null,
    "value": 3.1416312213178093
  },
  "smoke_opacity_weight": {
    "desc": null,
    "value": 0.7205311172210049
  },
  "checkpoint_iterations": {
    "desc": null,
    "value": [
      7000,
      15000,
      22000,
      30000
    ]
  },
  "depth_l1_weight_final": {
    "desc": null,
    "value": 0.2
  },
  "invalid_region_weight": {
    "desc": null,
    "value": 0.055431904572756846
  },
  "position_lr_max_steps": {
    "desc": null,
    "value": 30000
  },
  "densification_interval": {
    "desc": null,
    "value": 100
  },
  "opacity_reset_interval": {
    "desc": null,
    "value": 4500
  },
  "position_lr_delay_mult": {
    "desc": null,
    "value": 0.01
  },
  "cross_modal_edge_weight": {
    "desc": null,
    "value": 0.001
  },
  "densify_from_iter_stage2": {
    "desc": null,
    "value": 16000
  },
  "prune_more_opacity_ratio": {
    "desc": null,
    "value": 0.05
  },
  "densify_grad_threshold_smoke": {
    "desc": null,
    "value": 0.0003771229040189497
  },
  "smoke_uniformity_color_weight": {
    "desc": null,
    "value": 2.8740533789564067
  },
  "densify_grad_threshold_surface": {
    "desc": null,
    "value": 0.00009905924876138148
  },
  "prune_opacity_surface_threshold": {
    "desc": null,
    "value": 0.005
  }
}

command1 = f"python train_finetune_thermal.py --eval -s /home/neham/wildfire_all_data/real/drone_AFCA/align_rgb_thermal/bathroom/undistorted -m ./output/undistorted/20250216-19-07-18 --depths /home/neham/wildfire_all_data/real/drone_AFCA/align_rgb_thermal/bathroom/undistorted/images_depth"
command2 = f"python train_finetune_thermal.py --eval -s /home/neham/wildfire_all_data/real/drone_AFCA/align_rgb_thermal/red_container_colmap/undistorted -m ./output/undistorted/20250217-23-16-40 --depths /home/neham/wildfire_all_data/real/drone_AFCA/align_rgb_thermal/red_container_colmap/undistorted/images_depth"

for arg_name in [
    'smoke_opacity_weight',
    'smoke_color_weight',
    'densify_grad_threshold_surface',
    'densify_grad_threshold_smoke',
    'smoke_uniformity_color_weight',
    'thermal_weight',
    'depth_l1_weight_init',
    'invalid_region_weight',
    'valid_region_weight',
]:
    arg_value = args[arg_name]["value"] 
    command1 += f" --{arg_name}={arg_value}"
    command2 += f" --{arg_name}={arg_value}"

command1 += " --use_thermal"
command1 += " --use_wandb"
command2 += " --use_thermal"
command2 += " --use_wandb"



print(command1)
print("--------------------------------")
print(command2)
