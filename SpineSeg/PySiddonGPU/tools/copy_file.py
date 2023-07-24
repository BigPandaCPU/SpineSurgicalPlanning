import os
import shutil
from tqdm import tqdm

src_dir = "/media/alg/data3/DeepSpineData/CTSpine1k_new/drr_images/"
save_png_dir = "/media/alg/data3/DeepSpineData/drr_spine/drr_data/png"
os.makedirs(save_png_dir, exist_ok=True)

sub_dirs = os.listdir(src_dir)
for sub_dir in tqdm(sub_dirs):
    src_front_png_file = os.path.join(src_dir, sub_dir, "DRR_front.png")
    dst_front_png_file = os.path.join(save_png_dir, sub_dir+"_drr_front.png")

    src_side_png_file = os.path.join(src_dir, sub_dir, "DRR_side.png")
    dst_side_png_file = os.path.join(save_png_dir, sub_dir+"_drr_side.png")

    shutil.copy(src_front_png_file, dst_front_png_file)
    shutil.copy(src_side_png_file, dst_side_png_file)

