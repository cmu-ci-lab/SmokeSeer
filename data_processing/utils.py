import struct
import numpy as np
import collections
from scipy.spatial.transform import Rotation
import torch
import clip
from PIL import Image
from tqdm import tqdm
import os
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)

def match_rgb_thermal_images_using_clip(rgb_filenames, thermal_filenames, rgb_frames_path, thermal_frames_path, num_samples=40):
    """
    Match RGB and thermal images using CLIP features.
    
    Args:
        rgb_filenames: Dictionary mapping image IDs to filenames for RGB images
        thermal_filenames: Dictionary mapping image IDs to filenames for thermal images
        rgb_frames_path: Path to RGB image frames
        thermal_frames_path: Path to thermal image frames
        num_samples: Number of sample pairs to return
        
    Returns:
        sampled_rgb_ids: List of sampled RGB image IDs
        sampled_thermal_ids: List of corresponding thermal image IDs
    """
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Extract CLIP features for RGB images
    rgb_features = {}
    print("Extracting RGB features...")
    with torch.no_grad():
        for img_id, filename in tqdm(rgb_filenames.items()):
            img_path = os.path.join(rgb_frames_path, filename)
            try:
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                feature = model.encode_image(image)
                rgb_features[img_id] = feature.cpu().numpy()[0]
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Extract CLIP features for thermal images
    thermal_features = {}
    print("Extracting thermal features...")
    with torch.no_grad():
        for img_id, filename in tqdm(thermal_filenames.items()):
            img_path = os.path.join(thermal_frames_path, filename)
            try:
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                feature = model.encode_image(image)
                thermal_features[img_id] = feature.cpu().numpy()[0]
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Match RGB to thermal images
    rgb_ids = list(rgb_features.keys())
    thermal_ids = list(thermal_features.keys())
    
    rgb_feature_matrix = np.stack([rgb_features[img_id] for img_id in rgb_ids])
    thermal_feature_matrix = np.stack([thermal_features[img_id] for img_id in thermal_ids])
    
    # Normalize features
    rgb_feature_matrix = rgb_feature_matrix / np.linalg.norm(rgb_feature_matrix, axis=1, keepdims=True)
    thermal_feature_matrix = thermal_feature_matrix / np.linalg.norm(thermal_feature_matrix, axis=1, keepdims=True)
    
    # Compute similarity matrix
    similarity = np.dot(rgb_feature_matrix, thermal_feature_matrix.T)
    
    # Find best matches
    matches = {}
    for i, rgb_id in enumerate(rgb_ids):
        sim_scores = similarity[i]
        best_idx = np.argmax(sim_scores)
        matches[rgb_id] = thermal_ids[best_idx]
    
    # Sample matches for alignment
    num_samples = min(num_samples, len(matches))
    if len(matches) > num_samples:
        # Sample evenly across the dataset
        step = len(matches) // num_samples
        sampled_rgb_ids = list(matches.keys())[::step][:num_samples]
    else:
        sampled_rgb_ids = list(matches.keys())
    sampled_thermal_ids = [matches[rgb_id] for rgb_id in sampled_rgb_ids]
    
    return sampled_rgb_ids, sampled_thermal_ids

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_name] = {
                'id': image_id,
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'name': image_name,
                'xys': xys,
                'point3D_ids': point3D_ids
            }

    return images

def read_points3D_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = {
                'id': point3D_id,
                'xyz': xyz,
                'rgb': rgb,
                'error': error,
                'image_ids': image_ids,
                'point2D_idxs': point2D_idxs,
            }
    return points3D

def find_closest_keypoint(keypoints, x, y):
    dist = np.linalg.norm(keypoints - np.array([x, y]), axis=1)
    dist_idx = np.argmin(dist)
    return dist_idx, dist[dist_idx]

def umeyama(src, dst):
    assert src.shape == dst.shape
    n, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = np.sum(S) / np.trace(src_centered.T @ src_centered)
    t = dst_mean - scale * R @ src_mean
    return R, t, scale

def umeyama_ransac(src, dst, iterations=1000, threshold=0.05):
    """
    RANSAC version of Umeyama algorithm for robust transformation estimation.

    Args:
        src (np.ndarray): Source point cloud (N x dim).
        dst (np.ndarray): Destination point cloud (N x dim).
        iterations (int): Number of RANSAC iterations.
        threshold (float): Inlier distance threshold.

    Returns:
        tuple: (R, t, scale, inlier_mask) - Best transformation and inlier mask.
               Returns None if RANSAC fails to find a valid transformation.
    """
    assert src.shape == dst.shape
    n, dim = src.shape
    max_inliers = 0
    best_R, best_t, best_scale = None, None, None
    best_inlier_mask = None

    if n < dim + 1: # Need at least dim+1 points to estimate transformation
        R, t, scale = umeyama(src, dst)
        distances = np.linalg.norm(dst - (scale * (src @ R.T) + t), axis=1)
        inlier_mask = distances < threshold
        return R, t, scale, inlier_mask

    for _ in range(iterations):
        # Randomly sample minimal points (dim + 1 for affine, but using dim for simplicity and potentially better robustness)
        sample_indices = np.random.choice(n, size=min(dim, n), replace=False) # changed from dim+1 to dim
        src_sample = src[sample_indices]
        dst_sample = dst[sample_indices]

        # Estimate transformation using Umeyama
        try:
            R_sample, t_sample, scale_sample = umeyama(src_sample, dst_sample)
        except np.linalg.LinAlgError:
            continue # Skip if SVD fails (e.g., singular matrix)

        # Transform all source points and calculate distances to destination points
        transformed_src = scale_sample * (src @ R_sample.T) + t_sample
        distances = np.linalg.norm(dst - transformed_src, axis=1)
        inlier_mask = distances < threshold
        num_inliers = np.sum(inlier_mask)


        # Update best transformation if current model has more inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_R, best_t, best_scale = R_sample, t_sample, scale_sample
            best_inlier_mask = inlier_mask

    if best_R is not None:
        return best_R, best_t, best_scale, best_inlier_mask
    else:
        return None # RANSAC failed to find a good transformation


def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def save_cameras_txt(cameras_rgb, cameras_thermal, path_to_save):
    cameras = {}
    rgb_id_map = {}
    thermal_id_map = {}
    next_id = 1

    for old_id, camera in cameras_rgb.items():
        rgb_id_map[old_id] = next_id
        cameras[next_id] = camera._replace(id=next_id)
        next_id += 1
        
    for old_id, camera in cameras_thermal.items():
        thermal_id_map[old_id] = next_id
        cameras[next_id] = camera._replace(id=next_id)
        next_id += 1

    # Write combined cameras file in COLMAP format
    with open(path_to_save, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        
        for camera in cameras.values():
            params_str = " ".join(map(str, camera.params))
            f.write(f"{camera.id} {camera.model} {camera.width} {camera.height} {params_str}\n")

    print("RGB camera ID mapping:", rgb_id_map)
    print("Thermal camera ID mapping:", thermal_id_map)

    return rgb_id_map, thermal_id_map

def save_images_txt(images_rgb, images_thermal, rgb_camera_id_map, thermal_camera_id_map, rgb_point_map, thermal_point_map, path_to_save, thermal_transform=None):
    """
    Merge images from RGB and thermal reconstructions, reassign unique image IDs, update the point3D_ids using the
    provided point mapping dictionaries and (if provided) transform the thermal poses to the RGB camera frame.
    The file contains, for each image, the camera pose on one line followed by its POINTS2D on subsequent lines.
    Returns:
      (rgb_img_map, thermal_img_map): Mappings from original image IDs to new ones.
    """
    merged_images = {}
    rgb_img_map = {}
    thermal_img_map = {}
    next_id = 1
    for name, img in images_rgb.items():
        new_id = next_id
        next_id += 1
        rgb_img_map[img['id']] = new_id
        new_point3D_ids = []
        for pid in img['point3D_ids']:
            if pid == -1:
                new_point3D_ids.append(-1)
            else:
                new_point3D_ids.append(rgb_point_map[pid])
        new_img = img.copy()
        new_img['id'] = new_id
        new_img['camera_id'] = rgb_camera_id_map[img['camera_id']]
        new_img['point3D_ids'] = np.array(new_point3D_ids)
        merged_images[new_id] = new_img

    for name, img in images_thermal.items():
        new_id = next_id
        next_id += 1
        thermal_img_map[img['id']] = new_id
        new_point3D_ids = []
        for pid in img['point3D_ids']:
            if pid == -1:
                new_point3D_ids.append(-1)
            else:
                new_point3D_ids.append(thermal_point_map[pid])
        new_img = img.copy()
        new_img['id'] = new_id
        new_img['camera_id'] = thermal_camera_id_map[img['camera_id']]
        # If a transformation is provided, transform the thermal camera pose
        if thermal_transform is not None:
            R_t, t_t, s_t = thermal_transform
            
            # Get thermal camera pose
            R_cam = qvec2rotmat(img['qvec'])  # World-to-camera rotation
            t_cam = img['tvec']
            # Get camera center and transform it to RGB world
            C_thermal = -R_cam.T @ t_cam
            C_rgb = s_t * (R_t @ C_thermal) + t_t
            R_new = R_cam @ R_t.T
            t_new = -R_new @ C_rgb
            qvec_new = rotmat2qvec(R_new)
            new_img['qvec'] = qvec_new
            new_img['tvec'] = t_new

        new_img['point3D_ids'] = np.array(new_point3D_ids)
        merged_images[new_id] = new_img

    with open(path_to_save, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img_id in sorted(merged_images.keys()):
            im = merged_images[img_id]
            qvec_str = " ".join(map(str, im['qvec'].tolist()))
            tvec_str = " ".join(map(str, im['tvec'].tolist()))
            f.write(f"{im['id']} {qvec_str} {tvec_str} {im['camera_id']} {im['name']}\n")
            points2d_str = " ".join([f"{xy[0]} {xy[1]} {pt_id}" for xy, pt_id in zip(im['xys'], im['point3D_ids'])])
            f.write(f"{points2d_str}\n")
    return rgb_img_map, thermal_img_map

def save_points_txt(points_rgb, points_thermal, path_to_save, thermal_transform=None):
    """
    Merge 3D points from RGB and thermal reconstructions, applying an optional transformation
    to the thermal points (so that they are in the same coordinate frame as the RGB).
    Reassign unique point IDs, and write them into a text file using COLMAP format.
    Each line contains:
      POINT3D_ID, X, Y, Z, R, G, B, ERROR, IMAGE_IDS, POINT2D_idxs
    Returns:
      (rgb_point_id_map, thermal_point_id_map): Mappings from the original point IDs to the new ones.
    """
    merged_points = {}
    rgb_point_id_map = {}
    thermal_point_id_map = {}
    next_id = 1

    # Process RGB points
    for old_id, pt in points_rgb.items():
        rgb_point_id_map[old_id] = next_id
        new_pt = pt.copy()
        new_pt['id'] = next_id
        merged_points[next_id] = new_pt
        next_id += 1

    for old_id, pt in points_thermal.items():
        thermal_point_id_map[old_id] = next_id
        new_pt = pt.copy()
        new_pt['id'] = next_id
        if thermal_transform is not None:
            R_t, t_t, s_t = thermal_transform
            new_pt['xyz'] = s_t * (R_t @ new_pt['xyz']) + t_t
        merged_points[next_id] = new_pt
        next_id += 1

    with open(path_to_save, "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, IMAGE_IDS, POINT2D_idxs\n")
        for pt_id, pt in merged_points.items():
            # Convert numpy arrays to lists for nicer formatting
            xyz_str = " ".join(map(str, pt['xyz'].tolist()))
            rgb_str = " ".join(map(str, pt['rgb'].tolist()))
            error_str = str(pt['error'])
            image_ids_str = " ".join(map(str, pt['image_ids'].tolist()))
            point2d_str = " ".join(map(str, pt['point2D_idxs'].tolist()))
            f.write(f"{pt_id} {xyz_str} {rgb_str} {error_str} {image_ids_str} {point2d_str}\n")

    return rgb_point_id_map, thermal_point_id_map
