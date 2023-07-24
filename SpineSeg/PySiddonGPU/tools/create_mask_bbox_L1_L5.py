import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

# label: 1: , 2:left hip, 3:right hip, 4:
aim_dir = "/media/alg/data3/DeepSpineData/Yolo_detect_L1_L5/drr_front_png"
data_dir = "/media/alg/data3/DeepSpineData/CTSpine1k_new/label26/good"
save_dir = "/media/alg/data3/DeepSpineData/Yolo_detect_L1_L5/bbox"
png_names = os.listdir(aim_dir)
label_names = [png_name.replace("_drr_front.png", "_seg.nii.gz") for png_name in png_names]

L1_label = 20

for label_name in tqdm(label_names):
    #sub_path = os.path.join(data_dir, s_name)

    label_file = os.path.join(data_dir, label_name)
    data_img = sitk.ReadImage(label_file)
    data_np = sitk.GetArrayFromImage(data_img)

    data_shape = data_np.shape
    data_X = data_shape[2]
    data_Y = data_shape[1]
    data_Z = data_shape[0]

    idx = np.where(data_np > (L1_label-1))
    minX = np.min(idx[2])
    maxX = np.max(idx[2])

    minY = np.min(idx[1])
    maxY = np.max(idx[1])

    minZ = np.min(idx[0])
    maxZ = np.max(idx[0])

    save_txt_file = os.path.join(save_dir, label_name.replace(".nii.gz", ".txt"))
    fp = open(save_txt_file, "w")
    fp.write("%.3f %.3f %.3f %.3f %.3f %.3f"%(minX/data_X, maxX/data_X, minY/data_Y, maxY/data_Y, minZ/data_Z, maxZ/data_Z))
    fp.close()
    print(save_txt_file, " done!")





