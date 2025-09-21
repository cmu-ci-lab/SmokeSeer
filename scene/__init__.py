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
import pdb
import os
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_thermal import GaussianSmokeThermalModel, GaussianSurfaceThermalModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
import copy
from typing import Union, Optional, List
class Scene:

    gaussians : GaussianModel

    def __init__(self, args: ModelParams, gaussians: Union[GaussianModel, GaussianSmokeThermalModel, GaussianSurfaceThermalModel], load_iteration: Optional[int] = None, shuffle: bool = True, resolution_scales: List[float] = [1.0], append: str = "", is_smoke: bool = False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if args.use_thermal:
            self.train_cameras_thermal = {}
            self.test_cameras_thermal = {}

        if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval,is_smoke=is_smoke, use_thermal=args.use_thermal)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval,is_smoke=is_smoke, use_thermal=args.use_thermal)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
                if args.use_thermal:
                    camlist.extend(scene_info.test_cameras_thermal)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
                if args.use_thermal:
                    camlist.extend(scene_info.train_cameras_thermal)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)


        self.cameras_extent = scene_info.nerf_normalization["radius"]
        try:
            self.train_cameras_unshuffled = {}
            self.test_cameras_unshuffled = {}

            self.train_cameras_thermal_unshuffled = {}
            self.test_cameras_thermal_unshuffled = {}
          
            self.train_cameras = torch.load(os.path.join(self.model_path, "train_cameras.pth"))
            self.test_cameras = torch.load(os.path.join(self.model_path, "test_cameras.pth"))
            if args.use_thermal:
                self.train_cameras_thermal = torch.load(os.path.join(self.model_path, "train_cameras_thermal.pth"))
                self.test_cameras_thermal = torch.load(os.path.join(self.model_path, "test_cameras_thermal.pth"))

            for resolution_scale in resolution_scales:
                self.train_cameras_unshuffled[resolution_scale] = sorted(self.train_cameras[resolution_scale], key=lambda x: x.uid)
                self.test_cameras_unshuffled[resolution_scale] = sorted(self.test_cameras[resolution_scale], key=lambda x: x.uid)
                if args.use_thermal:
                    self.train_cameras_thermal_unshuffled[resolution_scale] = sorted(self.train_cameras_thermal[resolution_scale], key=lambda x: x.uid)
                    self.test_cameras_thermal_unshuffled[resolution_scale] = sorted(self.test_cameras_thermal[resolution_scale], key=lambda x: x.uid)

            print("Successfully loaded cameras")
        except Exception as e:
            print(f"Error loading cameras: {e}")
            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
                if args.use_thermal:
                    self.train_cameras_thermal[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras_thermal, resolution_scale, args)
                    self.test_cameras_thermal[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras_thermal, resolution_scale, args)

            self.train_cameras_unshuffled = copy.deepcopy(self.train_cameras)
            self.test_cameras_unshuffled = copy.deepcopy(self.test_cameras)

            if args.use_thermal:
                self.train_cameras_thermal_unshuffled = copy.deepcopy(self.train_cameras_thermal)
                self.test_cameras_thermal_unshuffled = copy.deepcopy(self.test_cameras_thermal)


            for resolution_scale in resolution_scales:
                random.seed(42)
                random.shuffle(self.train_cameras[resolution_scale])
                random.shuffle(self.test_cameras[resolution_scale])
                if args.use_thermal:
                    random.seed(42)
                    random.shuffle(self.train_cameras_thermal[resolution_scale])
                    random.shuffle(self.test_cameras_thermal[resolution_scale])

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           f"point_cloud_{append}.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration,append="", gaussians=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        if gaussians is None:
            if "thermal" in append:
                self.gaussians.save_ply(os.path.join(point_cloud_path, f"point_cloud_{append}.ply"), is_thermal=True)
            else:
                self.gaussians.save_ply(os.path.join(point_cloud_path, f"point_cloud_{append}.ply"))
        else:
            if "thermal" in append:
                gaussians.save_ply(os.path.join(point_cloud_path, f"point_cloud_{append}.ply"), is_thermal=True)
            else:
                gaussians.save_ply(os.path.join(point_cloud_path, f"point_cloud_{append}.ply"))
        return os.path.join(point_cloud_path, f"point_cloud_{append}.ply")

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]
    
    def getTrainCamerasThermal(self, scale=1.0):
        return self.train_cameras_thermal[scale]
    
    def getTestCamerasThermal(self, scale=1.0):
        return self.test_cameras_thermal[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getTrainCamerasUnshuffled(self, scale=1.0):
        return self.train_cameras_unshuffled[scale]
    
    def getTestCamerasUnshuffled(self, scale=1.0):
        return self.test_cameras_unshuffled[scale]

    def getTrainCamerasThermalUnshuffled(self, scale=1.0):
        return self.train_cameras_thermal_unshuffled[scale]

    def getTestCamerasThermalUnshuffled(self, scale=1.0):
        return self.test_cameras_thermal_unshuffled[scale]

class Scene_Smoke:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0],append="",is_smoke=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval,is_smoke=is_smoke)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval,is_smoke=is_smoke)
        else:
            assert False, "Could not recognize scene type!"

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        # if self.loaded_iter:
        #     self.gaussians.load_ply(os.path.join(self.model_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(self.loaded_iter),
        #                                                    f"point_cloud_{append}.ply"))
        # else:
        #     self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent,initial_opacity=0.01)

    def save(self, iteration,append=""):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, f"point_cloud_{append}.ply"))
        return os.path.join(point_cloud_path, f"point_cloud_{append}.ply")

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]