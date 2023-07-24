import os
src_dir = "/media/alg/data3/DeepHipData/colon/"
sub_dir_names = os.listdir(src_dir)
fp = open("../data/aim_files.txt","w")

for sub_dir_name in sub_dir_names:
    bbox_path = os.path.join(src_dir, sub_dir_name, "bbox")
    dcm_path = os.path.join(src_dir, sub_dir_name, "DICOM")
    if os.path.exists(bbox_path):
        out_str = "%s\n"%dcm_path
        fp.write(out_str)

fp.close()

