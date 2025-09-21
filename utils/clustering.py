from scene import GaussianModel, GaussianSurfaceThermalModel, GaussianSmokeThermalModel
import torch
from utils.general_utils import inverse_sigmoid
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from utils.general_utils import inverse_sigmoid
from simple_knn._C import distCUDA2
from render import render_test
from itertools import combinations
import copy
import random
import pdb
import numpy as np
import wandb

def gmm_cluster_gaussians(all_features):
    """
    This function clusters the gaussians based on the features.
    """
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    cluster = gmm.fit_predict(all_features)
    cluster1 = cluster == 0
    cluster2 = cluster == 1
    return cluster1, cluster2

def kmeans_cluster_gaussians(all_features):
    """
    This function clusters the gaussians based on the features.
    """
    kmeans = KMeans(n_clusters=2, random_state=10).fit(all_features)
    cluster1 = kmeans.labels_ == 0
    cluster2 = kmeans.labels_ == 1
    return cluster1, cluster2

@torch.no_grad()
def separate_out_clusters(gaussians_smoke: GaussianModel, gaussians_surface: GaussianModel, deform, train_cameras, initialize_surface=True):
    """
    This function gets the smoke gaussians which have properties like  surface gaussians into separate clusters.
    """
    d_xyz_average = torch.zeros(gaussians_smoke.get_xyz.shape[0], device="cuda")
    d_opacity_average = torch.zeros(gaussians_smoke.get_xyz.shape[0], device="cuda")
    for camera in train_cameras:
        fid = camera.fid
        N = gaussians_smoke.get_xyz.shape[0]
        #Make fid into a tensor
        fid = torch.tensor([fid], device="cuda")
        time_input = fid.unsqueeze(0).expand(N, -1)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_smoke.get_xyz.detach(), time_input)
        #Calculate average value of d_xyz, d_rotation, d_scaling
        d_xyz_average += torch.abs(d_xyz).mean(dim=-1)
        d_opacity_average += torch.abs(d_opacity).mean(dim=-1)

    d_xyz_average = d_xyz_average / len(train_cameras)
    d_opacity_average = d_opacity_average / len(train_cameras)
    feature_color = gaussians_smoke._features_dc.squeeze()
    #Get variance of color across all color channels
    feature_color_var = feature_color.var(dim=-1).unsqueeze(1)
    feature_min_scale = gaussians_smoke.get_scaling.min(dim=-1).values.unsqueeze(1)

    all_features = torch.cat((d_xyz_average.unsqueeze(-1), d_opacity_average.unsqueeze(-1), feature_min_scale, feature_color, feature_color_var), dim=1).cpu().numpy()

    cluster1, cluster2 = gmm_cluster_gaussians(all_features)
    if d_xyz_average[cluster1].mean() > d_xyz_average[cluster2].mean():
        cluster1, cluster2 = cluster2, cluster1

    
    #Cluster 1 is a binary mask of the gaussians
    # num_points_to_transfer = int(0.4 * cluster2.sum())
    # cluster2 = torch.from_numpy(cluster2).to(device="cuda")
    # indices_to_transfer = torch.nonzero(cluster2).squeeze()
    # indices_to_transfer = indices_to_transfer[torch.randperm(indices_to_transfer.size(0))[:num_points_to_transfer]]
    # indices_to_transfer = indices_to_transfer.cpu().numpy()
    # cluster1[indices_to_transfer] = True
    #Make cluster1 all 1s
    if initialize_surface:
        cluster1 = torch.ones_like(torch.from_numpy(cluster1))

    gaussians_surface._xyz = gaussians_smoke._xyz[cluster1].detach().clone()
    gaussians_surface._features_dc = gaussians_smoke._features_dc[cluster1].detach().clone()
    gaussians_surface._features_rest = gaussians_smoke._features_rest[cluster1].detach().clone()
    gaussians_surface._scaling = gaussians_smoke._scaling[cluster1].detach().clone()
    gaussians_surface._rotation = gaussians_smoke._rotation[cluster1].detach().clone()
    gaussians_surface._opacity = gaussians_smoke._opacity[cluster1].detach().clone()
    gaussians_surface.max_radii2D = gaussians_smoke.max_radii2D[cluster1].detach().clone()
    
    gaussians_smoke._xyz = gaussians_smoke._xyz[cluster2].detach().clone() 
    gaussians_smoke._features_dc = gaussians_smoke._features_dc[cluster2].detach().clone()
    gaussians_smoke._features_rest = gaussians_smoke._features_rest[cluster2].detach().clone()
    gaussians_smoke._scaling = gaussians_smoke._scaling[cluster2].detach().clone()
    gaussians_smoke._rotation = gaussians_smoke._rotation[cluster2].detach().clone()
    gaussians_smoke._opacity = gaussians_smoke._opacity[cluster2].detach().clone()
    gaussians_smoke.max_radii2D = gaussians_smoke.max_radii2D[cluster2].detach().clone()
 
    initial_opacity = 0.1
    gaussians_smoke._opacity = inverse_sigmoid(initial_opacity * torch.ones((gaussians_smoke._opacity.shape[0], 1), dtype=torch.float, device="cuda"))
    gaussians_surface._opacity = inverse_sigmoid(initial_opacity * torch.ones((gaussians_surface._opacity.shape[0], 1), dtype=torch.float, device="cuda"))

@torch.no_grad()
def separate_out_clusters_thermal(gaussians_smoke: GaussianSmokeThermalModel, gaussians_surface: GaussianSurfaceThermalModel, deform, train_cameras, initialize_surface=True):
    """
    This function gets the smoke gaussians which have properties like  surface gaussians into separate clusters.
    """
    d_xyz_average = torch.zeros(gaussians_smoke.get_xyz.shape[0], device="cuda")
    d_opacity_average = torch.zeros(gaussians_smoke.get_xyz.shape[0], device="cuda")
    for camera in train_cameras:
        fid = camera.fid
        N = gaussians_smoke.get_xyz.shape[0]
        #Make fid into a tensor
        fid = torch.tensor([fid], device="cuda")
        time_input = fid.unsqueeze(0).expand(N, -1)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_smoke.get_xyz.detach(), time_input)
        #Calculate average value of d_xyz, d_rotation, d_scaling
        d_xyz_average += torch.abs(d_xyz).mean(dim=-1)
        d_opacity_average += torch.abs(d_opacity).mean(dim=-1)

    d_xyz_average = d_xyz_average / len(train_cameras)
    d_opacity_average = d_opacity_average / len(train_cameras)
    feature_color = gaussians_smoke._features_dc.squeeze()
    #Get variance of color across all color channels
    feature_color_var = feature_color.var(dim=-1).unsqueeze(1)
    feature_min_scale = gaussians_smoke.get_scaling.min(dim=-1).values.unsqueeze(1)

    feature_opacity_difference = gaussians_smoke.get_opacity - gaussians_smoke.get_opacity_thermal
    all_features = torch.cat((d_xyz_average.unsqueeze(-1), d_opacity_average.unsqueeze(-1), feature_min_scale, feature_color, feature_opacity_difference), dim=1).cpu().numpy()

    cluster1, cluster2 = gmm_cluster_gaussians(all_features)
    if d_xyz_average[cluster1].mean() > d_xyz_average[cluster2].mean():
        cluster1, cluster2 = cluster2, cluster1

    if initialize_surface:
        cluster1 = torch.ones_like(torch.from_numpy(cluster1))

    gaussians_surface._xyz = gaussians_smoke._xyz[cluster1].detach().clone()
    gaussians_surface._features_dc = gaussians_smoke._features_dc[cluster1].detach().clone()
    gaussians_surface._features_rest = gaussians_smoke._features_rest[cluster1].detach().clone()
    gaussians_surface._scaling = gaussians_smoke._scaling[cluster1].detach().clone()
    gaussians_surface._rotation = gaussians_smoke._rotation[cluster1].detach().clone()
    gaussians_surface._opacity = gaussians_smoke._opacity[cluster1].detach().clone()
    gaussians_surface._features_thermal_dc = gaussians_smoke._features_thermal_dc[cluster1].detach().clone()
    gaussians_surface._features_thermal_rest = gaussians_smoke._features_thermal_rest[cluster1].detach().clone()
    gaussians_surface.max_radii2D = gaussians_smoke.max_radii2D[cluster1].detach().clone()
    
    gaussians_smoke._xyz = gaussians_smoke._xyz[cluster2].detach().clone() 
    gaussians_smoke._features_dc = gaussians_smoke._features_dc[cluster2].detach().clone()
    gaussians_smoke._features_rest = gaussians_smoke._features_rest[cluster2].detach().clone()
    gaussians_smoke._scaling = gaussians_smoke._scaling[cluster2].detach().clone()
    gaussians_smoke._rotation = gaussians_smoke._rotation[cluster2].detach().clone()
    gaussians_smoke._opacity = gaussians_smoke._opacity[cluster2].detach().clone()
    gaussians_smoke._opacity_thermal = gaussians_smoke._opacity_thermal[cluster2].detach().clone()
    gaussians_smoke._features_thermal_dc = gaussians_smoke._features_thermal_dc[cluster2].detach().clone()
    gaussians_smoke._features_thermal_rest = gaussians_smoke._features_thermal_rest[cluster2].detach().clone()
    gaussians_smoke.max_radii2D = gaussians_smoke.max_radii2D[cluster2].detach().clone()

    initial_opacity = 0.1
    gaussians_smoke._opacity = inverse_sigmoid(initial_opacity * torch.ones((gaussians_smoke._opacity.shape[0], 1), dtype=torch.float, device="cuda"))
    gaussians_surface._opacity = inverse_sigmoid(initial_opacity * torch.ones((gaussians_surface._opacity.shape[0], 1), dtype=torch.float, device="cuda"))

@torch.no_grad()
def separate_out_clusters_test(gaussians_smoke: GaussianModel, gaussians_surface: GaussianModel, deform, train_cameras, initialize_surface=True, other_params=None):
    """
    This function gets the smoke gaussians which have properties like  surface gaussians into separate clusters.
    """
    d_xyz_average = torch.zeros(gaussians_smoke.get_xyz.shape[0], device="cuda")
    d_opacity_average = torch.zeros(gaussians_smoke.get_xyz.shape[0], device="cuda")
    for camera in train_cameras:
        fid = camera.fid
        N = gaussians_smoke.get_xyz.shape[0]
        #Make fid into a tensor
        fid = torch.tensor([fid], device="cuda")
        time_input = fid.unsqueeze(0).expand(N, -1)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_smoke.get_xyz.detach(), time_input)
        #Calculate average value of d_xyz, d_rotation, d_scaling
        d_xyz_average += torch.abs(d_xyz).mean(dim=-1)
        d_opacity_average += torch.abs(d_opacity).mean(dim=-1)

    d_xyz_average = d_xyz_average / len(train_cameras)
    d_opacity_average = d_opacity_average / len(train_cameras)
    feature_color = gaussians_smoke._features_dc.squeeze()
    #Get variance of color across all color channels
    feature_color_var = feature_color.var(dim=-1).unsqueeze(1)
    feature_min_scale = gaussians_smoke.get_scaling.min(dim=-1).values.unsqueeze(1)

    feats = [feature_min_scale.cpu().numpy(), feature_color.cpu().numpy(), feature_color_var.cpu().numpy()]
    all_features_list = []
    for r in range(0, len(feats) + 1):
        for subset in combinations(feats, r):
            #Use d_xyz_average and d_opacity_average as well in subset to get the best combination
            subset = list(subset)
            subset.append(d_xyz_average.unsqueeze(-1).cpu().numpy())
            subset.append(d_opacity_average.unsqueeze(-1).cpu().numpy())
            all_features_list.append(np.concatenate(subset, axis=1))

    # Assuming you want to use the last combination for clustering
    psnr_gts = []
    pipe, background, iteration, model_path = other_params
    train_cameras_og = copy.deepcopy(train_cameras)
    gaussians_smoke_og = copy.deepcopy(gaussians_smoke)
    gaussians_surface_og = copy.deepcopy(gaussians_surface)

    for all_features in all_features_list:
        cluster1, cluster2 = gmm_cluster_gaussians(all_features)
        if d_xyz_average[cluster1].mean() > d_xyz_average[cluster2].mean():
            cluster1, cluster2 = cluster2, cluster1

        if initialize_surface:
            cluster1 = torch.ones_like(torch.from_numpy(cluster1))

        gaussians_surface = copy.deepcopy(gaussians_surface_og)
        gaussians_smoke = copy.deepcopy(gaussians_smoke_og)

        gaussians_surface._xyz = gaussians_smoke._xyz[cluster1].detach().clone()
        gaussians_surface._features_dc = gaussians_smoke._features_dc[cluster1].detach().clone()
        gaussians_surface._features_rest = gaussians_smoke._features_rest[cluster1].detach().clone()
        gaussians_surface._scaling = gaussians_smoke._scaling[cluster1].detach().clone()
        gaussians_surface._rotation = gaussians_smoke._rotation[cluster1].detach().clone()
        gaussians_surface._opacity = gaussians_smoke._opacity[cluster1].detach().clone()
        gaussians_surface.max_radii2D = gaussians_smoke.max_radii2D[cluster1].detach().clone()
        
        gaussians_smoke._xyz = gaussians_smoke._xyz[cluster2].detach().clone() 
        gaussians_smoke._features_dc = gaussians_smoke._features_dc[cluster2].detach().clone()
        gaussians_smoke._features_rest = gaussians_smoke._features_rest[cluster2].detach().clone()
        gaussians_smoke._scaling = gaussians_smoke._scaling[cluster2].detach().clone()
        gaussians_smoke._rotation = gaussians_smoke._rotation[cluster2].detach().clone()
        gaussians_smoke._opacity = gaussians_smoke._opacity[cluster2].detach().clone()
        gaussians_smoke.max_radii2D = gaussians_smoke.max_radii2D[cluster2].detach().clone()

        train_cameras = random.sample(train_cameras_og, 20)
        psnr_gt = render_test(train_cameras, gaussians_smoke, gaussians_surface, pipe, background, iteration , model_path, deform)
        psnr_gts.append(psnr_gt)

    best_index = psnr_gts.index(max(psnr_gts))
    all_features = all_features_list[best_index]

    wandb.log({"psnr_gt_final": max(psnr_gts)}, step=iteration)
    wandb.log({"best_index": best_index}, step=iteration)
    
    psnr_gts_list = [[i, psnr_gt] for i, psnr_gt in enumerate(psnr_gts)]
    wandb.log({"psnr_gts_all": wandb.Table(data=psnr_gts_list, columns=["index", "psnr_gt"])}, step=iteration)

    #Do a rendering with the best features
    cluster1, cluster2 = gmm_cluster_gaussians(all_features)
    if d_xyz_average[cluster1].mean() > d_xyz_average[cluster2].mean():
        cluster1, cluster2 = cluster2, cluster1

    if initialize_surface:
        cluster1 = torch.ones_like(torch.from_numpy(cluster1))

    gaussians_surface = copy.deepcopy(gaussians_surface_og)
    gaussians_smoke = copy.deepcopy(gaussians_smoke_og)
    
    gaussians_surface._xyz = gaussians_smoke._xyz[cluster1].detach().clone()
    gaussians_surface._features_dc = gaussians_smoke._features_dc[cluster1].detach().clone()
    gaussians_surface._features_rest = gaussians_smoke._features_rest[cluster1].detach().clone()
    gaussians_surface._scaling = gaussians_smoke._scaling[cluster1].detach().clone()
    gaussians_surface._rotation = gaussians_smoke._rotation[cluster1].detach().clone()
    gaussians_surface._opacity = gaussians_smoke._opacity[cluster1].detach().clone()
    gaussians_surface.max_radii2D = gaussians_smoke.max_radii2D[cluster1].detach().clone()
    
    gaussians_smoke._xyz = gaussians_smoke._xyz[cluster2].detach().clone() 
    gaussians_smoke._features_dc = gaussians_smoke._features_dc[cluster2].detach().clone()
    gaussians_smoke._features_rest = gaussians_smoke._features_rest[cluster2].detach().clone()
    gaussians_smoke._scaling = gaussians_smoke._scaling[cluster2].detach().clone()
    gaussians_smoke._rotation = gaussians_smoke._rotation[cluster2].detach().clone()
    gaussians_smoke._opacity = gaussians_smoke._opacity[cluster2].detach().clone()
    gaussians_smoke.max_radii2D = gaussians_smoke.max_radii2D[cluster2].detach().clone()

    psnr_gt = render_test(train_cameras, gaussians_smoke, gaussians_surface, pipe, background, iteration , model_path, deform, append="best_features")

    return max(psnr_gts), best_index

@torch.no_grad()
def separate_out_clusters_thermal_test(gaussians_smoke: GaussianModel, gaussians_surface: GaussianModel, deform, train_cameras, initialize_surface=True, other_params=None):
    """
    This function gets the smoke gaussians which have properties like  surface gaussians into separate clusters.
    """
    d_xyz_average = torch.zeros(gaussians_smoke.get_xyz.shape[0], device="cuda")
    d_opacity_average = torch.zeros(gaussians_smoke.get_xyz.shape[0], device="cuda")
    for camera in train_cameras:
        fid = camera.fid
        N = gaussians_smoke.get_xyz.shape[0]
        #Make fid into a tensor
        fid = torch.tensor([fid], device="cuda")
        time_input = fid.unsqueeze(0).expand(N, -1)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_smoke.get_xyz.detach(), time_input)
        #Calculate average value of d_xyz, d_rotation, d_scaling
        d_xyz_average += torch.abs(d_xyz).mean(dim=-1)
        d_opacity_average += torch.abs(d_opacity).mean(dim=-1)

    d_xyz_average = d_xyz_average / len(train_cameras)
    d_opacity_average = d_opacity_average / len(train_cameras)
    feature_color = gaussians_smoke._features_dc.squeeze()
    #Get variance of color across all color channels
    feature_color_var = feature_color.var(dim=-1).unsqueeze(1)
    feature_min_scale = gaussians_smoke.get_scaling.min(dim=-1).values.unsqueeze(1)

    feature_opacity_difference = gaussians_smoke.get_opacity - gaussians_smoke.get_opacity_thermal

    feats = [feature_min_scale.cpu().numpy(), feature_color.cpu().numpy(), feature_color_var.cpu().numpy()]
    all_features_list = []
    for r in range(0, len(feats) + 1):
        for subset in combinations(feats, r):
            #Use d_xyz_average and d_opacity_average as well in subset to get the best combination
            subset = list(subset)
            subset.append(d_xyz_average.unsqueeze(-1).cpu().numpy())
            subset.append(d_opacity_average.unsqueeze(-1).cpu().numpy())
            subset.append(feature_opacity_difference.cpu().numpy())
            all_features_list.append(np.concatenate(subset, axis=1))

    # Assuming you want to use the last combination for clustering
    psnr_gts = []
    pipe, background, iteration, model_path = other_params
    train_cameras_og = copy.deepcopy(train_cameras)
    gaussians_smoke_og = copy.deepcopy(gaussians_smoke)
    gaussians_surface_og = copy.deepcopy(gaussians_surface)

    for all_features in all_features_list:
        cluster1, cluster2 = gmm_cluster_gaussians(all_features)
        if d_xyz_average[cluster1].mean() > d_xyz_average[cluster2].mean():
            cluster1, cluster2 = cluster2, cluster1

        if initialize_surface:
            cluster1 = torch.ones_like(torch.from_numpy(cluster1))

        gaussians_surface = copy.deepcopy(gaussians_surface_og)
        gaussians_smoke = copy.deepcopy(gaussians_smoke_og)

        gaussians_surface._xyz = gaussians_smoke._xyz[cluster1].detach().clone()
        gaussians_surface._features_dc = gaussians_smoke._features_dc[cluster1].detach().clone()
        gaussians_surface._features_rest = gaussians_smoke._features_rest[cluster1].detach().clone()
        gaussians_surface._scaling = gaussians_smoke._scaling[cluster1].detach().clone()
        gaussians_surface._rotation = gaussians_smoke._rotation[cluster1].detach().clone()
        gaussians_surface._opacity = gaussians_smoke._opacity[cluster1].detach().clone()
        gaussians_surface._features_thermal_dc = gaussians_smoke._features_thermal_dc[cluster1].detach().clone()
        gaussians_surface._features_thermal_rest = gaussians_smoke._features_thermal_rest[cluster1].detach().clone()
        gaussians_surface.max_radii2D = gaussians_smoke.max_radii2D[cluster1].detach().clone()
        
        gaussians_smoke._xyz = gaussians_smoke._xyz[cluster2].detach().clone() 
        gaussians_smoke._features_dc = gaussians_smoke._features_dc[cluster2].detach().clone()
        gaussians_smoke._features_rest = gaussians_smoke._features_rest[cluster2].detach().clone()
        gaussians_smoke._scaling = gaussians_smoke._scaling[cluster2].detach().clone()
        gaussians_smoke._rotation = gaussians_smoke._rotation[cluster2].detach().clone()
        gaussians_smoke._opacity = gaussians_smoke._opacity[cluster2].detach().clone()
        gaussians_smoke._opacity_thermal = gaussians_smoke._opacity_thermal[cluster2].detach().clone()
        gaussians_smoke._features_thermal_dc = gaussians_smoke._features_thermal_dc[cluster2].detach().clone()
        gaussians_smoke._features_thermal_rest = gaussians_smoke._features_thermal_rest[cluster2].detach().clone()
        gaussians_smoke.max_radii2D = gaussians_smoke.max_radii2D[cluster2].detach().clone()

        train_cameras = random.sample(train_cameras_og, 20)
        psnr_gt = render_test(train_cameras, gaussians_smoke, gaussians_surface, pipe, background, iteration , model_path, deform)
        psnr_gts.append(psnr_gt)

    best_index = psnr_gts.index(max(psnr_gts))
    all_features = all_features_list[best_index]

    wandb.log({"psnr_gt_final": max(psnr_gts)}, step=iteration)
    wandb.log({"best_index": best_index}, step=iteration)
    
    psnr_gts_list = [[i, psnr_gt] for i, psnr_gt in enumerate(psnr_gts)]
    wandb.log({"psnr_gts_all": wandb.Table(data=psnr_gts_list, columns=["index", "psnr_gt"])}, step=iteration)

    #Do a rendering with the best features
    cluster1, cluster2 = gmm_cluster_gaussians(all_features)
    if d_xyz_average[cluster1].mean() > d_xyz_average[cluster2].mean():
        cluster1, cluster2 = cluster2, cluster1

    if initialize_surface:
        cluster1 = torch.ones_like(torch.from_numpy(cluster1))

    gaussians_surface = copy.deepcopy(gaussians_surface_og)
    gaussians_smoke = copy.deepcopy(gaussians_smoke_og)
    
    gaussians_surface._xyz = gaussians_smoke._xyz[cluster1].detach().clone()
    gaussians_surface._features_dc = gaussians_smoke._features_dc[cluster1].detach().clone()
    gaussians_surface._features_rest = gaussians_smoke._features_rest[cluster1].detach().clone()
    gaussians_surface._scaling = gaussians_smoke._scaling[cluster1].detach().clone()
    gaussians_surface._rotation = gaussians_smoke._rotation[cluster1].detach().clone()
    gaussians_surface._opacity = gaussians_smoke._opacity[cluster1].detach().clone()
    gaussians_surface._features_thermal_dc = gaussians_smoke._features_thermal_dc[cluster1].detach().clone()
    gaussians_surface._features_thermal_rest = gaussians_smoke._features_thermal_rest[cluster1].detach().clone()
    gaussians_surface.max_radii2D = gaussians_smoke.max_radii2D[cluster1].detach().clone()
    
    gaussians_smoke._xyz = gaussians_smoke._xyz[cluster2].detach().clone() 
    gaussians_smoke._features_dc = gaussians_smoke._features_dc[cluster2].detach().clone()
    gaussians_smoke._features_rest = gaussians_smoke._features_rest[cluster2].detach().clone()
    gaussians_smoke._scaling = gaussians_smoke._scaling[cluster2].detach().clone()
    gaussians_smoke._rotation = gaussians_smoke._rotation[cluster2].detach().clone()
    gaussians_smoke._opacity = gaussians_smoke._opacity[cluster2].detach().clone()
    gaussians_smoke._opacity_thermal = gaussians_smoke._opacity_thermal[cluster2].detach().clone()
    gaussians_smoke._features_thermal_dc = gaussians_smoke._features_thermal_dc[cluster2].detach().clone()
    gaussians_smoke._features_thermal_rest = gaussians_smoke._features_thermal_rest[cluster2].detach().clone()
    gaussians_smoke.max_radii2D = gaussians_smoke.max_radii2D[cluster2].detach().clone()

    psnr_gt = render_test(train_cameras, gaussians_smoke, gaussians_surface, pipe, background, iteration , model_path, deform, append="best_features")

    return max(psnr_gts), best_index