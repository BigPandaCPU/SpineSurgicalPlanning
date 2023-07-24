import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def draw_rectangle(draw, coordinates, color, width=1):
    """

    :param draw:
    :param coordinates:
    :param color:
    :param width:
    :return:
    """
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline = color)

def showData(data):
    """
    :param image:
    :return:
    """
    ans = plt.imshow(data, cmap=plt.cm.gray)
    plt.colorbar()
    plt.show()

if __name__=="__main__":
    import os
    from tqdm import tqdm
    drr_png_dir = "/media/alg/data3/DeepSpineData/drr_spine/drr_data/png"
    save_png_dir = "/media/alg/data3/DeepSpineData/drr_spine/drr_data/png_with_label"
    save_txt_dir = "/media/alg/data3/DeepSpineData/drr_spine/drr_data/yolo_label_txt"
    bbox_dir = "/media/alg/data3/DeepSpineData/drr_spine/drr_data/bbox"

    os.makedirs(save_png_dir, exist_ok=True)
    os.makedirs(save_txt_dir, exist_ok=True)

    drr_png_names = os.listdir(drr_png_dir)
    for drr_png_name in tqdm(drr_png_names):

        if "_front" in drr_png_name:
            continue

        print(drr_png_name)
        src_img_file = os.path.join(drr_png_dir, drr_png_name)
        img = Image.open(src_img_file)
        data = np.array(img)
        h, w = data.shape[0:2]

        bbox_file = os.path.join(bbox_dir, drr_png_name.replace("_drr_side.png", "_seg.txt"))
        minX, maxX, minY, maxY, minZ, maxZ = np.loadtxt(bbox_file)

        deltaY = 20
        deltaZ = 10

        dy = deltaY/w
        dz = deltaZ/h

        minY = np.max([minY-dy, 0.0])
        maxY = np.min([maxY+dy, 1.0])

        minZ = np.max([minZ-dz, 0.0])
        maxZ = np.min([maxZ+dz, 1.0])



        drawing = ImageDraw.Draw(img)
        top_left = (minY*w, (1.0 - maxZ)*h)  # (x, y)
        bottom_right = (maxY*w, (1.0 - minZ)*h) #

        label = 1
        center_X = (minY+maxY)/2.0
        center_Y = ((1.0-maxZ)+(1.0-minZ))/2.0

        dw = maxY - minY
        dh = (1.0 - minZ) - (1.0 - maxZ)

        save_label_file = os.path.join(save_txt_dir, drr_png_name.replace(".png", ".txt"))
        fp = open(save_label_file, "w")
        out_str = "%d %.4f %.4f %.4f %.4f\n"%(label, center_X, center_Y, dw, dh)
        fp.write(out_str)
        fp.close()


        save_img_file = os.path.join(save_png_dir, drr_png_name)
        top_left = ((center_X - dw/2.0)*w, (center_Y - dh/2.0)*h)  # (x, y)
        bottom_right = ((center_X + dw/2.0)*w, (center_Y + dh/2.0)*h) #

        outline_width = 2
        outline_color = "red"

        draw_rectangle(drawing, (top_left, bottom_right), color=outline_color, width=outline_width)
        img.save(save_img_file)
        #print("down")
        #break
