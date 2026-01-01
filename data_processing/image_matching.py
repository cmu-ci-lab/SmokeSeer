import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import os
from utils import *
import pdb
import wandb
import argparse
import open3d as o3d
from MINIMA.src.utils.load_model import load_model
import sys
import pycolmap

sys.path.append(os.path.join(os.path.dirname(__file__), 'MINIMA'))

def get_all_points(points):
    all_points = []
    all_colors = []
    for pid, pt in points.items():
        all_points.append(pt['xyz'])
        all_colors.append(pt['rgb'])
    return np.array(all_points), np.array(all_colors)


def main(args):
    # Load data
    images_rgb = read_images_binary(os.path.join(args.rgb_colmap_path, "images.bin"))
    points_rgb = read_points3D_binary(os.path.join(args.rgb_colmap_path, "points3D.bin"))
    images_thermal = read_images_binary(os.path.join(args.thermal_colmap_path, "images.bin"))
    points_thermal = read_points3D_binary(os.path.join(args.thermal_colmap_path, "points3D.bin"))

    #Initialize wandb
    
    wandb.init(project="wildfire_alignment_bathroom")
    all_rgb_points, all_rgb_colors = get_all_points(points_rgb)
    all_thermal_points, all_thermal_colors = get_all_points(points_thermal)

    #Save combined rgb and thermal points to 
    all_points = np.concatenate((all_thermal_points, all_rgb_points), axis=0)
    all_colors = np.concatenate((all_thermal_colors, all_rgb_colors), axis=0)
    points_colors = np.concatenate((all_points, all_colors), axis=1)
    point_cloud = wandb.Object3D(data_or_path=points_colors)
    wandb.log({"rgb_thermal_points_before": point_cloud})

    # Extract all image filenames from COLMAP data
    rgb_filenames = {}  # Map from image ID to filename
    thermal_filenames = {}  # Map from image ID to filename
    
    for img_id, img_data in images_rgb.items():
        rgb_filenames[img_id] = img_data['name']
    
    for img_id, img_data in images_thermal.items():
        thermal_filenames[img_id] = img_data['name']
    
    # Use CLIP-based feature matching to pair RGB and thermal images
    sampled_rgb_ids, sampled_thermal_ids = match_rgb_thermal_images_using_clip(
        rgb_filenames, 
        thermal_filenames, 
        args.rgb_frames, 
        args.thermal_frames, 
        num_samples=40
    )

    matcher = load_model("roma", args)
    src_pts, dst_pts = [], []

    for rgb_id, thermal_id in zip(sampled_rgb_ids, sampled_thermal_ids):
        rgb_filename = rgb_filenames[rgb_id]
        thermal_filename = thermal_filenames[thermal_id]
        
        path1 = os.path.join(args.rgb_frames, rgb_filename)
        path2 = os.path.join(args.thermal_frames, thermal_filename)

        matches = matcher(path1, path2)
        mkpts0 = matches['mkpts0']
        mkpts1 = matches['mkpts1']
        mconf = matches['mconf']

        mkpts0 = mkpts0[mconf > 0.7]
        mkpts1 = mkpts1[mconf > 0.7]
        mconf = mconf[mconf > 0.7]
        
        rgb_image_data = images_rgb.get(rgb_id)
        thermal_image_data = images_thermal.get(thermal_id)
        if not rgb_image_data or not thermal_image_data:
            continue
         
        for (x1, y1), (x2, y2) in zip(mkpts0, mkpts1):
            kp_rgb, dist_rgb = find_closest_keypoint(rgb_image_data['xys'], x1, y1)
            kp_thermal, dist_thermal = find_closest_keypoint(thermal_image_data['xys'], x2, y2)
            if dist_rgb > 1 or dist_thermal > 1:
                continue
            
            xyz_id_rgb = rgb_image_data['point3D_ids'][kp_rgb]
            xyz_id_thermal = thermal_image_data['point3D_ids'][kp_thermal]
            
            if xyz_id_rgb == -1 or xyz_id_thermal == -1:
                continue
            pt_rgb = points_rgb.get(xyz_id_rgb)['xyz']
            pt_thermal = points_thermal.get(xyz_id_thermal)['xyz']
            if pt_rgb is not None and pt_thermal is not None:
                src_pts.append(pt_thermal)
                dst_pts.append(pt_rgb)
    
    if len(src_pts) < 4:
        raise RuntimeError("Insufficient correspondences")

    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)
    R, t, s, inlier_mask = umeyama_ransac(src_pts, dst_pts)

    if len(src_pts) < 4:
        raise RuntimeError("Insufficient correspondences")

    # similarity_transform = pycolmap.estimate_sim3d_robust(src_pts, dst_pts)    
    
    # s = similarity_transform["tgt_from_src"].scale
    # R = similarity_transform["tgt_from_src"].rotation.matrix()
    # t = similarity_transform["tgt_from_src"].translation

    #Print error between R_old and R
    # print("Error between R_old and R: ", np.linalg.norm(R_old - R))
    # print("Error between t_old and t: ", np.linalg.norm(t_old - t))
    # print("Error between s_old and s: ", np.linalg.norm(s_old - s))

    transformed_thermal_points = s * (R @ all_thermal_points.T) + t.reshape(-1, 1)
    transformed_thermal_points = transformed_thermal_points.T
    #Save RGB and thermal points in the same ply file with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(transformed_thermal_points)
    pcd.colors = o3d.utility.Vector3dVector(all_thermal_colors)

    rgb_pcd = o3d.geometry.PointCloud()
    rgb_pcd.points = o3d.utility.Vector3dVector(all_rgb_points)
    rgb_pcd.colors = o3d.utility.Vector3dVector(all_rgb_colors)
    pcd += rgb_pcd  
    # Convert point cloud to numpy arrays for logging
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    points_colors = np.concatenate((points, colors), axis=1)
    point_cloud = wandb.Object3D(data_or_path=points_colors)
    wandb.log({"rgb_thermal_points": point_cloud})
    o3d.io.write_point_cloud("rgb_thermal_points.ply", pcd)

    cameras_rgb = read_cameras_binary(os.path.join(args.rgb_colmap_path, "cameras.bin"))
    cameras_thermal = read_cameras_binary(os.path.join(args.thermal_colmap_path, "cameras.bin"))
    os.makedirs(args.save_path, exist_ok=True)
    rgb_id_map, thermal_id_map = save_cameras_txt(cameras_rgb, cameras_thermal, os.path.join(args.save_path, "cameras.txt"))
    rgb_point_map, thermal_point_map = save_points_txt(points_rgb, points_thermal, os.path.join(args.save_path, "points3D.txt"), thermal_transform=(R, t, s))
    save_images_txt(images_rgb, images_thermal, rgb_id_map, thermal_id_map, rgb_point_map, thermal_point_map, os.path.join(args.save_path, "images.txt"), thermal_transform=(R, t, s))

    # Add new argument for undistorted path
    undistorted_path = os.path.join(os.path.dirname(args.save_path), "undistorted")
    os.makedirs(undistorted_path, exist_ok=True)

    #copy all rgb images and thermal images to a common path
    frames_path = os.path.join(os.path.dirname(args.save_path), "frames")
    os.makedirs(frames_path, exist_ok=True)
    os.system(f"cp {args.rgb_frames}/*.jpg {frames_path}/")
    os.system(f"cp {args.thermal_frames}/*.jpg {frames_path}/")

    os.system(f"colmap image_undistorter --image_path {frames_path} --input_path {args.save_path} --output_path {undistorted_path}")
    os.makedirs(os.path.join(undistorted_path, "sparse/0"), exist_ok=True)
    os.system(f"mv {undistorted_path}/sparse/*.bin {os.path.join(undistorted_path, 'sparse/0')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="roma")
    parser.add_argument('--ckpt', type=str, default='./MINIMA/weights/minima_roma.pth')
    parser.add_argument('--ckpt2', type=str, default='large')
    parser.add_argument('--save_path', type=str, default='./new_gt_red_container_all/0')
    parser.add_argument('--rgb_colmap_path', type=str, default='/home/neham/wildfire_all_data/real/drone_AFCA/align_rgb_thermal/new_gt_red_container/undistorted/sparse/0')
    parser.add_argument('--thermal_colmap_path', type=str, default='/home/neham/wildfire_all_data/real/drone_AFCA/red_container/only_rgb/glomap/cache/reconstruction/0')
    parser.add_argument('--rgb_frames', type=str, default='/home/neham/wildfire_all_data/real/drone_AFCA/align_rgb_thermal/new_gt_red_container/undistorted/images')
    parser.add_argument('--thermal_frames', type=str, default='/home/neham/wildfire_all_data/real/drone_AFCA/red_container/only_rgb/frames')
    args = parser.parse_args()
    main(args)
