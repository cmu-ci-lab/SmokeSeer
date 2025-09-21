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
from sklearn import metrics
import wandb
from sklearn.cluster import DBSCAN
import torch
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pdb
from scene import Scene, DeformModel, GaussianModel
from os import makedirs
from gaussian_renderer import render, render_surface_smoke
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import imageio
import copy
from utils.image_utils import psnr
import matplotlib.pyplot as plt
import cv2  

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians_surface = GaussianModel(dataset.sh_degree)
        scene_surface = Scene(dataset, gaussians_surface, load_iteration=iteration, shuffle=False,append="surface")

        deform = DeformModel(True, False)
        deform.load_weights(dataset.model_path)
        
        gaussians_smoke = GaussianModel(dataset.sh_degree)
        scene_smoke = Scene(dataset, gaussians_smoke, load_iteration=iteration, shuffle=False,append="smoke")
                
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        render_test(scene_smoke.getTestCameras(), gaussians_smoke, gaussians_surface, pipeline, background, iteration, dataset.model_path, deform=deform)
        
@torch.no_grad()
def render_test(test_cameras, gaussians_smoke, gaussians_surface, pipe, background, iteration, model_path, deform, append=None):
    #render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    images = []
    if append is not None:
        results_path = os.path.join(model_path, "test_results" + append)
    else:
        results_path = os.path.join(model_path, "test_results")
    makedirs(results_path, exist_ok=True)
    psnr_gt = 0.0
    common_W, common_H = -1, -1
    with torch.no_grad():
        for idx, view in enumerate(tqdm(test_cameras, desc="Rendering progress")):
            N = gaussians_smoke.get_xyz.shape[0]
            time_input = view.fid.unsqueeze(0).expand(N, -1)
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_smoke.get_xyz.detach(), time_input)
            deform_parameters = [d_xyz, d_rotation, d_scaling, d_opacity, d_color]
            rendering = render_surface_smoke(view, gaussians_surface, gaussians_smoke, pipe, background, deform_parameters=deform_parameters)["render"].clamp(0.0, 1.0)
            render_surface = render(view, gaussians_surface, pipe, background)["render"].clamp(0.0, 1.0)
            render_smoke = render(view, gaussians_smoke, pipe, background, deform_parameters=deform_parameters)["render"].clamp(0.0, 1.0)

            gt = view.original_image[0:3, :, :].cuda()
            if view.original_image_desmoked is not None:
                gt_desmoke = view.original_image_desmoked[0:3, :, :].cuda()
            else:
                gt_desmoke = view.original_image[0:3, :, :].cuda()

            psnr_gt += psnr(render_surface, gt_desmoke).mean().double()
            #concatenate the two images side by side
            concatenated_image =  torch.cat((gt, rendering, gt_desmoke, render_surface, render_smoke), dim=2)
            #Make to numpy and cpu and opencv format
            concatenated_image = concatenated_image.detach().cpu().numpy()
            concatenated_image = 255*concatenated_image.transpose(1, 2, 0)
        
            #concatenated_image = concatenated_image[...,::-1]
            concatenated_image = concatenated_image.astype('uint8')
            #downscale the image by a factor of 2
            #Make concatenated_image a common size for all images
            if common_W == -1 and common_H == -1:
                common_W, common_H = concatenated_image.shape[1], concatenated_image.shape[0]
            else:
                concatenated_image = cv2.resize(concatenated_image, (common_W, common_H))

            concatenated_image = cv2.resize(concatenated_image, (0,0), fx=0.5, fy=0.5)
            imageio.imwrite(os.path.join(results_path, '{0:05d}'.format(idx) + ".png"), concatenated_image)
    
            images.append(concatenated_image)
    
    psnr_gt /= len(test_cameras)
    #Save the images as a video of mp4 format and 30 fps
    if append is not None:    
        imageio.mimsave(f"/tmp/test_video_{append}.mp4", images, fps=6)
        try:
            wandb.log({"test_video"+append: wandb.Video(f"/tmp/test_video_{append}.mp4")})
        except Exception as e:  
            print(e)
            pass

    return psnr_gt


    
@torch.no_grad()
def render_test_thermal(test_cameras, test_cameras_thermal, gaussians_smoke, gaussians_surface, pipe, background, iteration, model_path, deform, append=None):
    #render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    images = []
    if append is not None:
        results_path = os.path.join(model_path, "test_results" + append)
    else:
        results_path = os.path.join(model_path, "test_results")
    makedirs(results_path, exist_ok=True)
    results_path_individual = os.path.join(model_path, "test_results_individual")
    makedirs(results_path_individual, exist_ok=True)
    psnr_gt = 0.0
    common_W, common_H = -1, -1
    with torch.no_grad():
        for idx, view in enumerate(tqdm(test_cameras, desc="Rendering progress")):
            view_thermal = test_cameras_thermal[idx]
            N = gaussians_smoke.get_xyz.shape[0]
            time_input = view.fid.unsqueeze(0).expand(N, -1)
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_smoke.get_xyz.detach(), time_input)
            deform_parameters = [d_xyz, d_rotation, d_scaling, d_opacity, d_color]
            rendering = render_surface_smoke(view, gaussians_surface, gaussians_smoke, pipe, background, deform_parameters=deform_parameters)["render"].clamp(0.0, 1.0)
            render_surface = render(view, gaussians_surface, pipe, background)["render"].clamp(0.0, 1.0)
            render_smoke = render(view, gaussians_smoke, pipe, background, deform_parameters=deform_parameters)["render"].clamp(0.0, 1.0)
            gt = view.original_image[0:3, :, :].cuda()
            view_thermal_gt = view_thermal.original_image[0:3, :, :].cuda()
            if view.original_image_desmoked is not None:
                gt_desmoke = view.original_image_desmoked[0:3, :, :].cuda()
            else:
                gt_desmoke = view.original_image[0:3, :, :].cuda()

            psnr_gt += psnr(render_surface, gt_desmoke).mean().double()
            #concatenate the two images side by side
            #Make all cats the size of view_thermal_gt
            gt_desmoke = torch.nn.functional.interpolate(gt_desmoke.unsqueeze(0), size=(view_thermal_gt.shape[1], view_thermal_gt.shape[2]), mode='bilinear', align_corners=False).squeeze(0)
            gt = torch.nn.functional.interpolate(gt.unsqueeze(0), size=(view_thermal_gt.shape[1], view_thermal_gt.shape[2]), mode='bilinear', align_corners=False).squeeze(0)
            rendering = torch.nn.functional.interpolate(rendering.unsqueeze(0), size=(view_thermal_gt.shape[1], view_thermal_gt.shape[2]), mode='bilinear', align_corners=False).squeeze(0)
            render_surface = torch.nn.functional.interpolate(render_surface.unsqueeze(0), size=(view_thermal_gt.shape[1], view_thermal_gt.shape[2]), mode='bilinear', align_corners=False).squeeze(0)
            render_smoke = torch.nn.functional.interpolate(render_smoke.unsqueeze(0), size=(view_thermal_gt.shape[1], view_thermal_gt.shape[2]), mode='bilinear', align_corners=False).squeeze(0)

            concatenated_image =  torch.cat((gt_desmoke, gt, view_thermal_gt, rendering, render_surface, render_smoke), dim=2)
            #Make to numpy and cpu and opencv format
            concatenated_image = concatenated_image.detach().cpu().numpy()
            concatenated_image = 255*concatenated_image.transpose(1, 2, 0)
        
            #concatenated_image = concatenated_image[...,::-1]
            concatenated_image = concatenated_image.astype('uint8')
            #downscale the image by a factor of 2
            #Make concatenated_image a common size for all images
            if common_W == -1 and common_H == -1:
                common_W, common_H = concatenated_image.shape[1], concatenated_image.shape[0]
            else:
                concatenated_image = cv2.resize(concatenated_image, (common_W, common_H))

            #concatenated_image = cv2.resize(concatenated_image, (0,0), fx=0.5, fy=0.5)
            imageio.imwrite(os.path.join(results_path, '{0:05d}'.format(idx) + ".png"), concatenated_image)
    
            images.append(concatenated_image)
 
            #Save all images individually also with viewpoint.image_name in individual
            for i, image in enumerate([gt_desmoke, gt, view_thermal_gt, rendering, render_surface, render_smoke]):
                image = image.detach().cpu().numpy()
                image = 255*image.transpose(1, 2, 0)
                image = image.astype('uint8')
                imageio.imwrite(os.path.join(results_path_individual, '{0:05d}'.format(idx) + "_{0:05d}".format(i) + ".png"), image)

    
    psnr_gt /= len(test_cameras)
    #Save the images as a video of mp4 format and 30 fps
    if append is not None:    
        imageio.mimsave(f"/tmp/test_video_{append}.mp4", images, fps=6)
        try:
            wandb.log({"test_video"+append: wandb.Video(f"/tmp/test_video_{append}.mp4")})
        except Exception as e:  
            print(e)
            pass

    return psnr_gt

def render_less_covariance(test_cameras, gaussians_surface, pipe, background, deform):
    #Calculate average deformation rotation scaling
        scaling = gaussians_surface.get_scaling     
        #Get min/max scaling for each gaussian scaling is (N, 3)
        min_scaling = scaling.min(dim=-1).values
        max_scaling = scaling.max(dim=-1).values
        #Calculate covariance
        all_covariance  = min_scaling
        all_covariance_numpy = all_covariance[all_covariance<0.01].cpu().numpy()
        plt.figure()
        plt.hist(all_covariance_numpy, bins=100, alpha=0.5, label='covariance')
        plt.savefig("histogram_covariance.png")
        #Get gaussians less than the mean deformation
        less_than_mean_covariance = all_covariance <0.006

        gaussians_surface._xyz = gaussians_surface._xyz[less_than_mean_covariance]
        gaussians_surface._features_dc = gaussians_surface._features_dc[less_than_mean_covariance]
        gaussians_surface._features_rest = gaussians_surface._features_rest[less_than_mean_covariance]
        gaussians_surface._scaling = gaussians_surface._scaling[less_than_mean_covariance]
        gaussians_surface._rotation = gaussians_surface._rotation[less_than_mean_covariance]
        gaussians_surface._opacity = gaussians_surface._opacity[less_than_mean_covariance]

        view = test_cameras[0]
        N = gaussians_surface.get_xyz.shape[0]
        time_input = view.fid.unsqueeze(0).expand(N, -1)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_surface.get_xyz.detach(), time_input)
        deform_parameters =  [d_xyz, d_rotation, d_scaling, d_opacity, d_color]
        
        image = render(view, gaussians_surface, pipe, background, deform_parameters=deform_parameters)["render"].clamp(0.0, 1.0)
        image = image.cpu().numpy().transpose(1, 2, 0)
        #TypeError: Cannot handle this data type: (1, 1, 3), <f4
        image = (image * 255).astype("uint8")
        imageio.imwrite("render_less_covariance.png", image)

def render_less_smoke(test_cameras, gaussians_surface, pipe, background, deform):    
    #Calculate average deformation rotation scaling
    d_xyz_average = torch.zeros(gaussians_surface.get_xyz.shape[0], device="cuda")
    with torch.no_grad():
        for fid in range(100):
            fid = fid/100
            N = gaussians_surface.get_xyz.shape[0]
            #Make fid into a tensor
            fid = torch.tensor([fid], device="cuda")
            time_input = fid.unsqueeze(0).expand(N, -1)
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_surface.get_xyz.detach(), time_input)
            #Calculate average value of d_xyz, d_rotation, d_scaling
            d_xyz_average += torch.abs(d_xyz).mean(dim=-1)

        view = test_cameras[0]
        N = gaussians_surface.get_xyz.shape[0]
        time_input = view.fid.unsqueeze(0).expand(N, -1)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_surface.get_xyz.detach(), time_input)
        deform_parameters =  [d_xyz, d_rotation, d_scaling, d_opacity, d_color]
        
        image = render(view, gaussians_surface, pipe, background, deform_parameters=deform_parameters)["render"].clamp(0.0, 1.0)
        image = image.cpu().numpy().transpose(1, 2, 0)
        #TypeError: Cannot handle this data type: (1, 1, 3), <f4
        image = (image * 255).astype("uint8")
        imageio.imwrite("render_more_smoke.png", image)
        d_xyz_average = d_xyz_average / 100

        all_deformations = d_xyz_average
        
        #Get gaussians less than the mean deformation
        less_than_mean_deformation = all_deformations < 0.0001

        gaussians_surface._xyz = gaussians_surface._xyz[less_than_mean_deformation]
        gaussians_surface._features_dc = gaussians_surface._features_dc[less_than_mean_deformation]
        gaussians_surface._features_rest = gaussians_surface._features_rest[less_than_mean_deformation]
        gaussians_surface._scaling = gaussians_surface._scaling[less_than_mean_deformation]
        gaussians_surface._rotation = gaussians_surface._rotation[less_than_mean_deformation]
        gaussians_surface._opacity = gaussians_surface._opacity[less_than_mean_deformation]

        d_xyz_average = d_xyz_average.cpu().numpy()

        plt.figure()
        plt.hist(d_xyz_average, bins=100, alpha=0.5, label='d_xyz')
        plt.savefig("histogram1.png")
        #Use the above for indexing the gaussian model
        view = test_cameras[0]
        N = gaussians_surface.get_xyz.shape[0]
        time_input = view.fid.unsqueeze(0).expand(N, -1)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_surface.get_xyz.detach(), time_input)
        deform_parameters =  [d_xyz, d_rotation, d_scaling, d_opacity, d_color]
        
        image = render(view, gaussians_surface, pipe, background, deform_parameters=deform_parameters)["render"].clamp(0.0, 1.0)
        image = image.cpu().numpy().transpose(1, 2, 0)
        #TypeError: Cannot handle this data type: (1, 1, 3), <f4
        image = (image * 255).astype("uint8")
        imageio.imwrite("render_less_smoke.png", image)

@torch.no_grad()
def render_clusters(test_cameras, gaussians_surface, pipe, background, deform):
    #Calculate average deformation rotation scaling
    # plt.figure()
    # plt.hist(all_covariance_numpy, bins=100, alpha=0.5, label='covariance')
    # plt.savefig("histogram_clusters.png")
    # #Get gaussians less than the mean deformation
    d_xyz_average = torch.zeros(gaussians_surface.get_xyz.shape[0], device="cuda") 
    with torch.no_grad():
        for fid in range(100):
            fid = fid/100
            N = gaussians_surface.get_xyz.shape[0]
            #Make fid into a tensor
            fid = torch.tensor([fid], device="cuda")
            time_input = fid.unsqueeze(0).expand(N, -1)
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_surface.get_xyz.detach(), time_input)
            #Calculate average value of d_xyz, d_rotation, d_scaling
            d_xyz_average += torch.abs(d_xyz).mean(dim=-1)
    
    feature_deformation = d_xyz_average.unsqueeze(1) / 100
    feature_color = gaussians_surface._features_dc.squeeze()
    #Get variance of color across all color channels
    feature_color_var = feature_color.var(dim=-1).unsqueeze(1)
    feature_min_scale = gaussians_surface.get_scaling.min(dim=-1).values.unsqueeze(1)
    all_features = torch.cat((feature_deformation, feature_color, feature_min_scale, feature_color_var), dim=1).cpu().numpy()

    all_features = StandardScaler().fit_transform(all_features)
    # Try to do divide them in two clusters based on these features  
    db = DBSCAN(eps=0.6, min_samples=1000).fit(all_features)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("Number of clusters: ", n_clusters_)
    n_noise_ = list(labels).count(-1)
    cluster1 = labels == 0
    cluster2 = labels == 1
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    #Call 1 cluster gaussians_surface and another gaussians_smoke (create copy of them)
    gaussians_smoke = copy.deepcopy(gaussians_surface)

    gaussians_surface._xyz = gaussians_surface._xyz[cluster1]
    gaussians_surface._features_dc = gaussians_surface._features_dc[cluster1]
    gaussians_surface._features_rest = gaussians_surface._features_rest[cluster1]
    gaussians_surface._scaling = gaussians_surface._scaling[cluster1]
    gaussians_surface._rotation = gaussians_surface._rotation[cluster1]
    gaussians_surface._opacity = gaussians_surface._opacity[cluster1]

    gaussians_smoke._xyz = gaussians_smoke._xyz[cluster2]
    gaussians_smoke._features_dc = gaussians_smoke._features_dc[cluster2]
    gaussians_smoke._features_rest = gaussians_smoke._features_rest[cluster2]
    gaussians_smoke._scaling = gaussians_smoke._scaling[cluster2]
    gaussians_smoke._rotation = gaussians_smoke._rotation[cluster2]
    gaussians_smoke._opacity = gaussians_smoke._opacity[cluster2]

    view = test_cameras[0]

    N = gaussians_surface.get_xyz.shape[0]

    time_input = view.fid.unsqueeze(0).expand(N, -1)
    d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_surface.get_xyz.detach(), time_input)

    deform_parameters =  [d_xyz, d_rotation, d_scaling, d_opacity, d_color]

    image_surface = render(view, gaussians_surface, pipe, background, deform_parameters=deform_parameters)["render"].clamp(0.0, 1.0)
    image_surface = image_surface.cpu().numpy().transpose(1, 2, 0)
    image_surface = (image_surface * 255).astype("uint8")
    imageio.imwrite("render_cluster_surface.png", image_surface)

    N = gaussians_smoke.get_xyz.shape[0]

    time_input = view.fid.unsqueeze(0).expand(N, -1)
    d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_smoke.get_xyz.detach(), time_input)

    deform_parameters =  [d_xyz, d_rotation, d_scaling, d_opacity, d_color]
    image_smoke = render(view, gaussians_smoke, pipe, background, deform_parameters=deform_parameters)["render"].clamp(0.0, 1.0)
    image_smoke = image_smoke.cpu().numpy().transpose(1, 2, 0)
    image_smoke = (image_smoke * 255).astype("uint8")
    imageio.imwrite("render_cluster_smoke.png", image_smoke)
       
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

