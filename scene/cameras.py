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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import cv2
import pdb
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", desmoked_image=None, desmoked_alpha_mask=None, fid = None,
                 smoke_mask = None, invdepthmap = None, depth_params = None, resolution = None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.focal_length = 0.5 * image.shape[2] / np.tan(0.5 * FoVx)

        try:
            self.data_device = torch.device("cpu")
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        if desmoked_image is not None:
            self.original_image_desmoked = desmoked_image.clamp(0.0, 1.0).to(self.data_device).to(dtype=torch.float16)
        else:
            self.original_image_desmoked = None
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.fid = torch.Tensor(np.array([fid])).cuda()

        if gt_alpha_mask is not None:
            self.alpha_mask = gt_alpha_mask.to(torch.uint8)
        else:
            self.alpha_mask = gt_alpha_mask

        if smoke_mask is not None:
            self.smoke_mask = smoke_mask.to(torch.float16)
        else:
            self.smoke_mask = smoke_mask
        
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            if desmoked_image is not None:
                self.original_image_desmoked *= desmoked_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            if desmoked_image is not None:
                self.original_image_desmoked *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.original_image).to(torch.uint8)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform_og = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        #Make a copy so that changes to self.world_view_transform do not affect self.world_view_transform_og
        self.world_view_transform = self.world_view_transform_og.clone().detach()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        #K is the intrinsic matrix
        self.K = torch.zeros((1, 3, 3), device=self.data_device)
        self.K[0, 0, 0] = 0.5 * self.image_width / np.tan(0.5 * self.FoVx)  # fx
        self.K[0, 1, 1] = 0.5 * self.image_height / np.tan(0.5 * self.FoVy) # fy  
        self.K[0, 0, 2] = self.image_width / 2  # cx
        self.K[0, 1, 2] = self.image_height / 2 # cy
        self.K[0, 2, 2] = 1.0

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

