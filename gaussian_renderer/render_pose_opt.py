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
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from gsplat.rendering import rasterization
import pdb
def render_pose_opt(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, deform_parameters=None, is_thermal=False, camera_opt=None):
    """
    Render the scene with optional deformation parameters.
    
    Background tensor (bg_color) must be on GPU!
    """
    
    from diff_gaussian_rasterization_pose import GaussianRasterizationSettings, GaussianRasterizer
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if camera_opt:
        world_view_transform = camera_opt(viewpoint_camera.world_view_transform.cuda(), viewpoint_camera.uid)
        full_proj_transform = viewpoint_camera.world_view_transform @ viewpoint_camera.projection_matrix
        camera_center = world_view_transform.inverse()[3, :3]
    else:
        world_view_transform = viewpoint_camera.world_view_transform
        full_proj_transform = viewpoint_camera.full_proj_transform
        camera_center = viewpoint_camera.camera_center

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
            viewmatrix=world_view_transform,
            perspectivematrix=viewpoint_camera.projection_matrix, # Added for computing gradients for viewmatrix
            projmatrix=full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if deform_parameters:
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform_parameters
        means3D = pc.get_xyz + d_xyz
        if is_thermal:
            opacity = pc.get_opacity_thermal
        else:
            opacity = pc.opacity_activation(pc._opacity + d_opacity)
        scales = pc.scaling_activation(pc._scaling + d_scaling)
        rotations = pc.rotation_activation(pc._rotation + d_rotation)
        if override_color is not None:
            colors = d_color  # [N, 3]
            sh_degree = None
        else:
            if is_thermal:
                colors = pc.get_features_thermal + d_color  # [N, K, 3]
            else:
                colors = pc.get_features + d_color  # [N, K, 3]
            sh_degree = pc.active_sh_degree
        cov3D_precomp = None
    else:
        means3D = pc.get_xyz
        if is_thermal:
            opacity = pc.get_opacity_thermal
        else:
            opacity = pc.get_opacity
        scales = pc.get_scaling * scaling_modifier
        rotations = pc.get_rotation
        if override_color is not None:
            colors = override_color  # [N, 3]
            sh_degree = None
        else:
            if is_thermal:
                colors = pc.get_features_thermal  # [N, K, 3]
            else:
                colors = pc.get_features  # [N, K, 3]
            sh_degree = pc.active_sh_degree
        cov3D_precomp = None

    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if is_thermal:
            shs = pc.get_features_thermal
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        viewmat=world_view_transform, # Added for receiving gradients
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    ret =  {
        "render": rendered_image.clamp(0, 1),
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

    return ret

def render_pose_opt_gsplat(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, deform_parameters=None, is_thermal=False, camera_opt=None):
    """
    Render the scene. 
    
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

    if deform_parameters:
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform_parameters
        means3D = pc.get_xyz + d_xyz
        if is_thermal:
            opacity = pc.get_opacity_thermal
        else:
            opacity = pc.opacity_activation(pc._opacity + d_opacity)
        scales = pc.scaling_activation(pc._scaling + d_scaling)
        rotations = pc.rotation_activation(pc._rotation + d_rotation)
        cov3D_precomp = None
        if is_thermal:
            colors = pc.get_features_thermal + d_color  # [N, K, 3]
        else:
            colors = pc.get_features + d_color  # [N, K, 3]
    else:
        means3D = pc.get_xyz
        if is_thermal:
            opacity = pc.get_opacity_thermal
        else:
            opacity = pc.get_opacity

        cov3D_precomp = None
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if is_thermal:
            colors = pc.get_features_thermal  # [N, K, 3]
        else:
            colors = pc.get_features  # [N, K, 3]

    sh_degree = pc.active_sh_degree
    if camera_opt:
        viewmat = viewpoint_camera.world_view_transform_og.transpose(0, 1) # [4, 4]
        viewmat = camera_opt(viewmat, viewpoint_camera.uid)
        viewpoint_camera.world_view_transform = viewmat.transpose(0, 1).detach().clone()
    else:
        viewmat = viewpoint_camera.world_view_transform_og.transpose(0, 1) # [4, 4]

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
    )
    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": info["means2d"],
            "visibility_filter" : radii > 0,
            "radii": radii}