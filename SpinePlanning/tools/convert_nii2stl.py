import os

import numpy as np
from tools.vtk_tools import *

def convert_nii2stl(src_label_dir, dst_stl_dir):
    file_names = os.listdir(src_label_dir)
    for cur_file_name in file_names:
        shot_name, ext = os.path.splitext(cur_file_name)
        cur_file_path = os.path.join(src_label_dir, cur_file_name)
        img_array, origin, spacing, direction = getImageFromNII(cur_file_path)
        unique_values = np.unique(img_array)

        # print(origin)
        # print(spacing)
        # print(direction)
        # print(unique_values)

        tmp_array = np.zeros_like(img_array)
        for i in range(len(unique_values)):
            cur_value = unique_values[i]
            tmp_array[:] = 0
            if cur_value > 0:
                cur_idx = np.where(img_array == cur_value)
                tmp_array[cur_idx] = cur_value
                save_stl_file_path = os.path.join(dst_stl_dir, shot_name+"_label_%02d.stl"%cur_value)
                cur_polydata_normal, cur_polydata = createPolyDataNormalsFromArray(tmp_array, spacing, origin)

                featureEdges = vtk.vtkFeatureEdges()
                featureEdges.FeatureEdgesOff()
                featureEdges.BoundaryEdgesOn()
                featureEdges.NonManifoldEdgesOn()
                featureEdges.SetInputData(cur_polydata)
                featureEdges.Update()

                numberOfOpenEdges = featureEdges.GetOutput().GetNumberOfCells()
                # if numberOfOpenEdges > 0:
                #     print("The label is partially clipped because of the size of dicom, "
                #           "coordinate creation for this kind of label might be incorrect")
                #     continue

                saveSTLFile(save_stl_file_path, cur_polydata_normal)
                print(save_stl_file_path, " saved done!")
        print("done")


if __name__ == '__main__':
    # data_path = 'G:\CTSpine1K-main\data\label'
    # dataset_name = os.listdir(data_path)
    # for dataset in dataset_name:
    #     if 'erse' in dataset:
    #         print('No verse dataset is needed')
    #         continue
    #     out_dir = os.path.join(data_path, dataset+'_stl')
    #     if not os.path.isdir(out_dir):
    #         os.mkdir(out_dir)
    #     dataset_dir = os.path.join(data_path, dataset)
    #     # nii_files = os.listdir(dataset_dir)
    #
    #     convert_nii2stl(dataset_dir, out_dir)
    #     # for nii_name in nii_files:
    #     #     nii_file_path = os.path.join(dataset_dir, nii_name)
    #     #     convert_nii2stl(nii_file_path, out_dir)
    #
    nii_dir = "D:\PointNet2_deepspine\\test_data\label"
    out_dir = "D:\PointNet2_deepspine\\test_data\stl"
    os.makedirs(out_dir, exist_ok=True)
    convert_nii2stl(nii_dir, out_dir)
