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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import pdb
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: int = 0
    gt_desmoked_image: np.array = None
    alpha_mask: np.array = None
    depth_params: dict = None
    depth_path: str = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    train_cameras_thermal: list = None
    test_cameras_thermal: list = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, use_thermal=False):
    cam_infos = []
    cam_infos_thermal = []
    max_frame_number = max([cam_extrinsics[key].frame_number for key in cam_extrinsics])
    min_frame_number = min([cam_extrinsics[key].frame_number for key in cam_extrinsics])
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()
        extr = cam_extrinsics[key]
        fid = (extr.frame_number - min_frame_number)/(max_frame_number - min_frame_number)
        print("frame number: ", extr.frame_number, "name: ", extr.name)
        print("fid: ", fid)
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        depth_params = None
        n_remove = len(extr.name.split('.')[-1]) + 1
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depth_params")
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        if "gt" in image_name:
            continue
        image = Image.open(image_path)
        #Append _gt to the image name
        gt_name = image_name + "_gt." + image_path.split(".")[1]
        gt_path = os.path.join(images_folder, gt_name)
        if os.path.exists(gt_path):
            gt_desmoked_image = Image.open(gt_path)
        else:
            gt_desmoked_image = None

        alpha_mask = None
        alpha_mask_path = os.path.join(images_folder + "_mask", image_name + ".png")

        if os.path.exists(alpha_mask_path):
            alpha_mask = Image.open(alpha_mask_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, 
                              gt_desmoked_image=gt_desmoked_image,fid=fid, alpha_mask=alpha_mask, 
                              depth_params=depth_params, depth_path=depth_path)

        if 'thermal' in image_name and use_thermal:
            cam_infos_thermal.append(cam_info)
        elif 'thermal' in image_name and not use_thermal:
            continue
        else:
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos, cam_infos_thermal

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, llffhold=8, is_smoke=False, use_thermal=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except Exception as e:
        print("Error reading binary files, trying to read text files")
        print(e)
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)


    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted, cam_infos_thermal_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params, 
                                                                       images_folder=os.path.join(path, reading_dir), 
                                                                       depths_folder=os.path.join(path, depths) if depths != "" else "", use_thermal=use_thermal)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    if use_thermal:
        cam_infos_thermal = sorted(cam_infos_thermal_unsorted.copy(), key = lambda x : x.image_name)
        if eval:
            train_cam_infos_thermal = [c for idx, c in enumerate(cam_infos_thermal) if idx % llffhold != 0]
            test_cam_infos_thermal = [c for idx, c in enumerate(cam_infos_thermal) if idx % llffhold == 0]
        else:
            train_cam_infos_thermal = cam_infos_thermal
            test_cam_infos_thermal = []
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    if not os.path.exists(ply_path) and is_smoke==False:
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        if is_smoke==False:
            pcd = fetchPly(ply_path)
        else:
            # num_pts = 100_000
            # print(f"Generating random point cloud ({num_pts})...")
            # # We create random points inside the bounds of the synthetic Blender scenes
            # xyz = np.random.random((num_pts, 3))
            # xyz[:, 0] = xyz[:, 0] * 5.0 - 2.5
            # xyz[:, 1] = xyz[:, 1] * 5.0 - 2.5
            # xyz[:, 2] = xyz[:, 2] * 6.0
            # shs = np.random.random((num_pts, 3)) / 255.0
            # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            pcd = fetchPly(ply_path)
    except:
        pcd = None

    if use_thermal:
        scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            train_cameras_thermal=train_cam_infos_thermal,
                            test_cameras_thermal=test_cam_infos_thermal,
                            ply_path=ply_path)
    else:   
        scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame["frame"]
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            cam_name = '/'.join(cam_name.split("/")[-2:])
            try:
                gt_cam_name = cam_name.split(".")[0] + "_gt." + cam_name.split(".")[1]
                if os.path.exists(os.path.join(path, gt_cam_name)):
                    gt_desmoked_image = Image.open(os.path.join(path, gt_cam_name))
                    gt_desmoked_image = np.array(gt_desmoked_image.convert("RGBA"))
                    bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                    norm_data = gt_desmoked_image / 255.0
                    arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                    gt_desmoked_image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")      
                else:
                    gt_desmoked_image = None
            except:
                gt_desmoked_image = None 

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], gt_desmoked_image=gt_desmoked_image,fid=frame_time))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png",is_smoke=False, use_thermal=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    
    if use_thermal:
        print("Reading Thermal Transforms")
        train_cam_infos_thermal = readCamerasFromTransforms(path, "transforms_thermaltrain.json", white_background, extension)
        test_cam_infos_thermal = readCamerasFromTransforms(path, "transforms_thermaltest.json", white_background, extension)
    else:
        train_cam_infos_thermal = []
        test_cam_infos_thermal = []
    
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if is_smoke==False:
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
    else:
        bin_path = os.path.join(path, "sparse/0/points3D_smoke.bin")

    if is_smoke==True:
        ply_path = os.path.join(path, "points3d_smoke.ply")
    else:
        ply_path = os.path.join(path, "points3d_surface.ply")
        
    os.system("rm -rf {}".format(ply_path))
    try:
        xyz, rgb, _ = read_points3D_binary(os.path.join(path, "sparse_colmap/points3D.bin"))
        storePly(ply_path, xyz, rgb)
    except:
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 3.2 - 1.6
        shs = np.random.random((num_pts, 3))
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           train_cameras_thermal=train_cam_infos_thermal,
                           test_cameras_thermal=test_cam_infos_thermal,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}