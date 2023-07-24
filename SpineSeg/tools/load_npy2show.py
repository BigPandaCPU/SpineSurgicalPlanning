import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sag_img = np.load("../sample/Test10/cor_img.npy")

save_png_file = "../sample/Test10/cor_img.png"

data = sag_img
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
