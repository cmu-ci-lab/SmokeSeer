folder = "/home/neham/wildfire_all_data/real/drone_AFCA/align_rgb_thermal/reconstruction_rgb_thermal/undistorted/images/thermal*"
import os
import glob
import cv2
import numpy as np
import pdb
import shutil

images = []
for file in glob.glob(folder):
    images.append(cv2.imread(file))

images = np.array(images)

images_variance = np.var(images, axis=-1)
images_variance = np.mean(images_variance, axis=0)

images_variance = np.where(images_variance > 30, 0, 255)

images_variance = images_variance.astype(np.uint8)
cv2.imwrite("mask.png", images_variance)
# # Create output directory if it doesn't exist
output_dir = "/home/neham/wildfire_all_data/real/drone_AFCA/align_rgb_thermal/reconstruction_rgb_thermal/undistorted/images_mask"
os.makedirs(output_dir, exist_ok=True)

# Copy mask.png to each thermal image location
for file in glob.glob(folder):
    file_name = os.path.basename(file).replace(".jpg", ".png")
    output_path = os.path.join(output_dir, file_name)
    shutil.copy("mask.png", output_path)
    #print(output_path)
