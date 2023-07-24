import os
import open3d as o3d
from vtkmodules.util import numpy_support
from tools.vtk_tools import *

if __name__ == "__main__":
    #point_file_dir = "E:/data/DeepSpineData/picked_points"
    stl_dir = "E:/data/DeepSpineData/stl"
    stl_file_names = os.listdir(stl_dir)
    for stl_file_name in stl_file_names:
        if "1.3.6.1.4.1.9328.50.4.0002_seg.nii_label" not in stl_file_name:
            continue
        print("\n"+stl_file_name)
        stl_file_path = os.path.join(stl_dir, stl_file_name)
        spine_axis_info, spine_points_info, all_actors = createSpineInfoFromStl(stl_file_path)

        showActors(all_actors)
