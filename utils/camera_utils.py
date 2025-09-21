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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import pdb
import cv2
import torch
WARNED = False


def smooth_binary_mask(binary_mask):
    # Apply distance transform
    binary_mask = np.array(binary_mask)
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    dist_transform = dist_transform / (binary_mask.shape[0] + binary_mask.shape[1])
    alpha = 20.0
    smoothed_mask = 1 - np.exp(-alpha * dist_transform)
    # #Visualize smoothed mask
    # smoothed_mask_viz = (smoothed_mask * 255).astype(np.uint8)
    # cv2.imwrite("smoothed_mask.png", smoothed_mask_viz)
    # pdb.set_trace()
    return smoothed_mask

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    if cam_info.gt_desmoked_image is not None:
        resized_image_gt_RGB = PILtoTorch(cam_info.gt_desmoked_image, resolution)
        resized_image_gt_RGB = resized_image_gt_RGB[:3, ...]
    else:
        resized_image_gt_RGB = None

    gt_image = resized_image_rgb[:3, ...]
    smoke_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
        loaded_mask_gt = resized_image_gt_RGB[3:4, ...]
    elif cam_info.alpha_mask is not None and "thermal" in cam_info.image_name:
        loaded_mask = PILtoTorch(cam_info.alpha_mask, resolution)
        loaded_mask_gt = PILtoTorch(cam_info.alpha_mask, resolution)
    elif cam_info.alpha_mask is not None and "rgb" in cam_info.image_name:
        smoke_mask = smooth_binary_mask(cam_info.alpha_mask)
        smoke_mask = torch.from_numpy(smoke_mask).unsqueeze(0).to(args.data_device)
        #Transform smoke mask to binary mask
        loaded_mask = None
        loaded_mask_gt = None
    else:
        loaded_mask = None
        loaded_mask_gt = None

    if cam_info.depth_path != "" and cam_info.depth_path is not None:
        try:
            invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)
        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, desmoked_image=resized_image_gt_RGB,desmoked_alpha_mask=loaded_mask_gt,fid=cam_info.fid, smoke_mask=smoke_mask, invdepthmap=invdepthmap, depth_params=cam_info.depth_params, resolution=resolution)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
