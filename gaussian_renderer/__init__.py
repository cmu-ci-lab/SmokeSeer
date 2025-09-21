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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_thermal import GaussianSurfaceThermalModel, GaussianSmokeThermalModel
from utils.sh_utils import eval_sh
import pdb
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, deform_parameters=None, is_thermal=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    #deform_parameters = None
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if deform_parameters:
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform_parameters
        means3D = pc.get_xyz + d_xyz
        means2D = screenspace_points
        if is_thermal:
            opacity = pc.get_opacity_thermal
        else:
            opacity = pc.get_opacity_at_t(viewpoint_camera.fid)
        scales = pc.scaling_activation(pc._scaling + d_scaling)
        rotations = pc.rotation_activation(pc._rotation + d_rotation)
        cov3D_precomp = None
    else:
        means3D = pc.get_xyz
        means2D = screenspace_points
        if is_thermal:
            opacity = pc.get_opacity_thermal
        else:
            opacity = pc.get_opacity

        cov3D_precomp = None
        scales = pc.get_scaling
        rotations = pc.get_rotation

    if is_thermal:
        rendered_image, radii, depth = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = pc.get_features_thermal,
                colors_precomp = None,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp   
        )
    else:
        rendered_image, radii, depth = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = pc.get_features,
                colors_precomp = None,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image.clamp(0, 1),   
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "inv_depth": depth}

def render_surface_smoke(viewpoint_camera, pc1 : GaussianSurfaceThermalModel, pc2 : GaussianSmokeThermalModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, deform_parameters = None, scale_divide = 1.0, is_thermal=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    #deform_parameters = None
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(torch.cat([pc1.get_xyz,pc2.get_xyz],dim=0), dtype=pc1.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc1.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if deform_parameters:
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform_parameters
        means3D = torch.cat([pc1.get_xyz,pc2.get_xyz+d_xyz],dim=0)
        means2D = screenspace_points
        if is_thermal:
            opacity = torch.cat([pc1.get_opacity_thermal,pc2.get_opacity_thermal],dim=0)
        else:
            opacity = torch.cat([pc1.get_opacity, pc2.get_opacity_at_t(viewpoint_camera.fid)],dim=0)
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = torch.cat([pc1.get_covariance(scaling_modifier),pc2.get_covariance(scaling_modifier)],dim=0)
        else:
            scales = torch.cat([pc1.get_scaling,pc2.scaling_activation(pc2._scaling+d_scaling)],dim=0)
            rotations = torch.cat([pc1.get_rotation,pc2.rotation_activation(pc2._rotation+d_rotation)],dim=0)
    else:
        means3D = torch.cat([pc1.get_xyz,pc2.get_xyz],dim=0)
        means2D = screenspace_points
        if is_thermal:
            opacity = torch.cat([pc1.get_opacity_thermal,pc2.get_opacity_thermal],dim=0)
        else:
            opacity = torch.cat([pc1.get_opacity,pc2.get_opacity],dim=0)
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = torch.cat([pc1.get_covariance(scaling_modifier),pc2.get_covariance(scaling_modifier)],dim=0)
        else:
            scales = torch.cat([pc1.get_scaling,pc2.get_scaling],dim=0)
            rotations = torch.cat([pc1.get_rotation,pc2.get_rotation],dim=0)

    if is_thermal:
        rendered_image, radii, depth = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = torch.cat([pc1.get_features_thermal,pc2.get_features_thermal],dim=0),
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )
    else:
        rendered_image, radii, depth = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = torch.cat([pc1.get_features,pc2.get_features],dim=0),
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image.clamp(0, 1),
            "visibility_filter_surface" : radii[:pc1.get_xyz.shape[0]] > 0,
            "radii_surface": radii[:pc1.get_xyz.shape[0]],
            "visibility_filter_smoke" : radii[pc1.get_xyz.shape[0]:] > 0,
            "radii_smoke": radii[pc1.get_xyz.shape[0]:],
            "viewspace_points": screenspace_points,
            "inv_depth": depth}
