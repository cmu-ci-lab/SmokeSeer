# Import the plyfile module
import plyfile
import numpy as np
import pdb
# Define the paths of the two ply files
path1 = "/home/neham/Desktop/gaussian-splatting/output/ficus_smoke_with_gt_video_new_0.9/20231230-13-21-48/point_cloud/iteration_20000/point_cloud.ply"
path2 = "/home/neham/Desktop/gaussian-splatting/output/ficus_smoke_with_gt_video_new_0.9/20231230-13-21-48/point_cloud/iteration_20000/point_cloud.ply"

# Read the ply files as PlyData objects
ply1 = plyfile.PlyData.read(path1)
ply2 = plyfile.PlyData.read(path2)

# Get the vertex elements from the ply files
vertex1 = ply1["vertex"]
vertex2 = ply2["vertex"]

# Create a list to store the combined attributes
combined = []
dtype_full = []
# Loop through the attributes of the vertex elements
for attribute in vertex1.properties:
    # Concatenate the attribute values from both ply files
    values = np.concatenate((vertex1[attribute.name], vertex2[attribute.name]), axis=0)
    # Append the attribute name and values to the combined list
    combined.append(values[:,np.newaxis])
    dtype_full.append((attribute.name, 'f4'))

pdb.set_trace()
attributes = np.concatenate(combined, axis=1)
elements = np.empty(combined[0].shape[0], dtype=dtype_full)
elements[:] = list(map(tuple, attributes))
# Create a new vertex element with the combined attributes
vertex = plyfile.PlyElement.describe(elements, "vertex")

# Create a new ply file with the new vertex element
ply = plyfile.PlyData([vertex])
ply.write("combined.ply")
