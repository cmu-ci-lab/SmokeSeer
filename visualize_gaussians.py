#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import pdb
from scene import Scene, Scene_Smoke, GaussianModel, DeformModel
from tqdm import tqdm
from gaussian_renderer import render, render_surface_smoke
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
def render_sets(dataset : ModelParams, iteration : int, pipe : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians_surface = GaussianModel(dataset.sh_degree)
        scene_surface = Scene(dataset, gaussians_surface, load_iteration=iteration, shuffle=False,append="surface")

        deform = DeformModel(False, False)
        deform.load_weights(dataset.model_path)
        images = []
        gaussians_smoke = GaussianModel(dataset.sh_degree)
        scene_smoke = Scene_Smoke(dataset, gaussians_smoke, load_iteration=iteration, shuffle=False,append="smoke")
        
        print(dataset.source_path)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        with torch.no_grad():
            for idx, view in enumerate(tqdm(scene_surface.getTestCameras(), desc="Rendering progress")):
                N = gaussians_smoke.get_xyz.shape[0]
                time_input = view.fid.unsqueeze(0).expand(N, -1)
                # pdb.set_trace()
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_smoke.get_xyz.detach(), time_input)
                deform_parameters = [d_xyz, d_rotation, d_scaling, d_opacity, d_color]

                #Gaussian smoke parameters
                gaussian_smoke_xyz = gaussians_smoke.get_xyz + d_xyz
                gaussian_smoke_scaling = gaussians_smoke.get_scaling + d_scaling
                gaussian_smoke_rotation = gaussians_smoke.get_rotation + d_rotation

                scale_ratio = torch.max(gaussian_smoke_scaling, dim=1).values/torch.min(gaussian_smoke_scaling, dim=1).values

                #Plot histogram of the scale ratio
                plt.figure()
                plt.hist(np.sort(scale_ratio.cpu().numpy())[5000:-5000], bins=1000)
                plt.title("Histogram of scale ratio")
                plt.savefig("histogram.png")
                pdb.set_trace()

                mask = torch.ones(N, dtype=torch.bool, device="cuda")
                new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation = gaussians_smoke.get_xyz
                new_xyz += d_xyz[mask]
                new_scaling += d_scaling[mask]
                new_rotation += d_rotation[mask]
                new_opacities += d_opacity[mask]
                rendering = render_surface_smoke(view, gaussians_surface, gaussians_smoke, pipe, background, deform_parameters=deform_parameters)["render"]
                gaussians_surface.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,new_rotation)
                render_surface = render(view, gaussians_surface, pipe, background)
                render_smoke = render(view, gaussians_smoke, pipe, background, deform_parameters=deform_parameters)

                gt = view.original_image[0:3, :, :]
                if view.original_image_desmoked is not None:
                    gt_desmoke = view.original_image_desmoked[0:3, :, :]
                else:
                    gt_desmoke = view.original_image[0:3, :, :]
                #concatenate the two images side by side
                concatenated_image =  torch.cat((gt, rendering, gt_desmoke, render_surface["render"], render_smoke["render"]), dim=2)
                #Make to numpy and cpu and opencv format
                concatenated_image = concatenated_image.detach().cpu().numpy()
                concatenated_image = 255*concatenated_image.transpose(1, 2, 0)
            
                #concatenated_image = concatenated_image[...,::-1]
                concatenated_image = concatenated_image.astype('uint8')
                #downscale the image by a factor of 2
                concatenated_image = cv2.resize(concatenated_image, (0,0), fx=0.5, fy=0.5)
                #Save the image
                images.append(concatenated_image)

        imageio.mimsave(f"test_video_examine_2.mp4", images, fps=12)
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

# python train_stage1.py -s /home/neham/mast3r/red_container_sky_segmented_undistorted/ --eval  --opacity_weight=1.6907221742870466 --deformation_weight=0.7953870393266802 --densify_grad_threshold_surface=0.0004173019788974 --use_thermal --pose_opt --use_wandb
# python train_stage1.py -s /home/neham/mast3r/red_container_sky_segmented_undistorted/ --eval  --opacity_weight=1.6907221742870466 --deformation_weight=0.7953870393266802 --densify_grad_threshold_surface=0.0004173019788974 --use_thermal --use_wandb
# python train_stage1.py -s /home/neham/mast3r/red_container_sky_segmented_undistorted_2/ --eval  --opacity_weight=1.6907221742870466 --deformation_weight=0.7953870393266802 --densify_grad_threshold_surface=0.0004173019788974 --use_thermal --pose_opt --use_wandb
# python train_stage1.py -s /home/neham/mast3r/red_container_sky_segmented_undistorted_2/ --eval  --opacity_weight=1.6907221742870466 --deformation_weight=0.7953870393266802 --densify_grad_threshold_surface=0.0004173019788974 --use_thermal --use_wandb