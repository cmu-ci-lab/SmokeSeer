import os
import glob
from natsort import natsorted

import cv2

files = natsorted(glob.glob("/home/neham/Desktop/gaussian_desmoking/output/undistorted_red_container/20241107-17-42-43/test_resultsfinal/*"))


# Define the codec and create VideoWriter object
frame = cv2.imread(files[0])
height, width, layers = frame.shape
video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 4, (width, height))

# Resize all images to the same size
for i in range(len(files)):
    img = cv2.imread(files[i])
    resized_img = cv2.resize(img, (width, height))
    video.write(resized_img)

cv2.destroyAllWindows()
video.release()