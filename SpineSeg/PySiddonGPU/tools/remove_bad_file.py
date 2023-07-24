import os
import shutil
src_dir = "/media/alg/data3/DeepSpineData/drr_spine/drr_data/bbox"
delete_dir = "/media/alg/data3/DeepSpineData/drr_spine/drr_data/yolo_label_txt"

yolo_text_png_dir = "/media/alg/data3/DeepSpineData/drr_spine/labels/valid"
os.makedirs(yolo_text_png_dir, exist_ok=True)

del_names = os.listdir(delete_dir)

for del_name in del_names:
    del_file = os.path.join(delete_dir, del_name)


    if "side" in del_name:
        src_file = os.path.join(src_dir, del_name[:-13]+"_seg.txt")
    else:
        src_file = os.path.join(src_dir, del_name[:-14] + "_seg.txt")
    #print(src_file)
    if os.path.exists(src_file):
        continue
    else:
        src_png_file = del_file
        dst_png_file = os.path.join(yolo_text_png_dir, del_name)
        shutil.move(src_png_file, dst_png_file)
        print(del_name)
