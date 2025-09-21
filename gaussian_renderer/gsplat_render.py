# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use 
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #

import math

import torch
from torch.nn import functional as F
from gsplat import rasterization
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_thermal import GaussianSurfaceThermalModel, GaussianSmokeThermalModel
import pdb

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, deform_parameters=None, is_thermal=False):
    """
    Render the scene using gsplat backend. 
    
    Background tensor (bg_color) must be on GPU!
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    # Handle deformation parameters if provided
    if deform_parameters:
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform_parameters
        means3D = pc.get_xyz + d_xyz
        scales = pc.scaling_activation(pc._scaling + d_scaling) * scaling_modifier
        rotations = pc.rotation_activation(pc._rotation + d_rotation)
        if is_thermal:
            opacity = pc.get_opacity_thermal
            colors = pc.get_features_thermal
        else:
            opacity = pc.get_opacity_at_t(viewpoint_camera.fid)
            colors = pc.get_features
    else:
        means3D = pc.get_xyz
        scales = pc.get_scaling * scaling_modifier
        rotations = pc.get_rotation
        if is_thermal:
            opacity = pc.get_opacity_thermal
            colors = pc.get_features_thermal
        else:
            opacity = pc.get_opacity
            colors = pc.get_features

    if override_color is not None:
        colors = override_color
        sh_degree = None
    else:
        sh_degree = pc.active_sh_degree

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)
    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
        render_mode="RGB+ED"
    )

    rendered_image_depth = render_colors[0] 
    radii = info["radii"].squeeze(0) 
    rendered_image = rendered_image_depth[:, :, :3].permute(2, 0, 1)
    depth = rendered_image_depth[:, :, 3]
    inv_depth = 1.0 / (depth + 1e-6)
    render_alphas = render_alphas.squeeze(0).permute(2, 0, 1)
    
    try:
        info["means2d"].retain_grad()
    except:
        pass

    return {
        "render": rendered_image.clamp(0, 1),
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "inv_depth": inv_depth, 
        "render_alphas": render_alphas
    }

def render_surface_smoke(viewpoint_camera, pc1 : GaussianSurfaceThermalModel, pc2 : GaussianSmokeThermalModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, deform_parameters = None, scale_divide = 1.0, is_thermal=False):
    """
    Render both surface and smoke gaussians using gsplat backend.
    
    Background tensor (bg_color) must be on GPU!
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    # Handle deformation parameters
    if deform_parameters:
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform_parameters
        means3D = torch.cat([pc1.get_xyz, pc2.get_xyz + d_xyz], dim=0)
        scales = torch.cat([
            pc1.get_scaling,
            pc2.scaling_activation(pc2._scaling + d_scaling)
        ], dim=0) * scaling_modifier
        rotations = torch.cat([
            pc1.get_rotation,
            pc2.rotation_activation(pc2._rotation + d_rotation)
        ], dim=0)
        if is_thermal:
            opacity = torch.cat([pc1.get_opacity_thermal, pc2.get_opacity_thermal], dim=0)
            colors = torch.cat([pc1.get_features_thermal, pc2.get_features_thermal], dim=0)
        else:
            opacity = torch.cat([
                pc1.get_opacity,
                pc2.get_opacity_at_t(viewpoint_camera.fid)
            ], dim=0)
            colors = torch.cat([pc1.get_features, pc2.get_features], dim=0)
    else:
        means3D = torch.cat([pc1.get_xyz, pc2.get_xyz], dim=0)
        scales = torch.cat([pc1.get_scaling, pc2.get_scaling], dim=0) * scaling_modifier
        rotations = torch.cat([pc1.get_rotation, pc2.get_rotation], dim=0)
        if is_thermal:
            opacity = torch.cat([pc1.get_opacity_thermal, pc2.get_opacity_thermal], dim=0)
            colors = torch.cat([pc1.get_features_thermal, pc2.get_features_thermal], dim=0)
        else:
            opacity = torch.cat([pc1.get_opacity, pc2.get_opacity], dim=0)
            colors = torch.cat([pc1.get_features, pc2.get_features], dim=0)

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)
    render_colors, render_alphas, info = rasterization(
        means=means3D,
        quats=rotations,
        scales=scales,
        opacities=opacity.squeeze(-1),
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=pc1.active_sh_degree,
        render_mode="RGB+D"
    )

    rendered_image_depth = render_colors[0]
    
    rendered_image = rendered_image_depth[:, :, :3].permute(2, 0, 1)
    depth = rendered_image_depth[:, :, 3]
    inv_depth = 1.0 / (depth + 1e-6)
    radii = info["radii"].squeeze(0)
    n_surface = pc1.get_xyz.shape[0]
    render_alphas = render_alphas.squeeze(0).permute(2, 0, 1)

    try:
        info["means2d"].retain_grad()
    except:
        pass

    return {
        "render": rendered_image.clamp(0, 1),
        "visibility_filter_surface": radii[:n_surface] > 0,
        "radii_surface": radii[:n_surface],
        "visibility_filter_smoke": radii[n_surface:] > 0,
        "radii_smoke": radii[n_surface:],
        "viewspace_points": info["means2d"],
        "inv_depth": inv_depth,
        "render_alphas": render_alphas
    }


