import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def convert_drr_txt2png(drr_txt_file, save_png_file):
    """

    :param dtt_txt_file:
    :param save_png_file:
    :return:
    """
    data = np.loadtxt(drr_txt_file)
    data=data[::-1, :]
    data_min = np.min(data)
    data_max = np.max(data)
    data = (data-data_min)/(data_max-data_min)*255

    image_out = Image.fromarray(data.astype('uint8'))
    img_as_img = image_out.convert("RGB")
    img_as_img.save(save_png_file)
    # ans = plt.imshow(data, cmap=plt.cm.gray)
    # plt.colorbar()
    # plt.show()
    print("down")


if __name__=="__main__":
    import os
    input_dir = "/media/alg/data3/DeepHipData/hip_test/dt10/predict"
    front_drr_file = os.path.join(input_dir, "dt10_front.txt")
    front_png_file = os.path.join(input_dir, "dt10_front.png")
    convert_drr_txt2png(front_drr_file, front_png_file)

    side_drr_file = os.path.join(input_dir, "dt10_side.txt")
    side_png_file = os.path.join(input_dir, "dt10_side.png")
    convert_drr_txt2png(side_drr_file, side_png_file)

