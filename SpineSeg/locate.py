__copyright__ = \
"""
Copyright &copyright © (c) 2021 Inria Grenoble Rhône-Alpes.
All rights reserved.

This source code is to be used for academic research purposes only.
For commercial uses of the code, please send an email to edmond.boyer@inria.fr and sergi.pujades@inria.fr

"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Di Meng"



def get_norm_sagittal_coronal_images(pir_vol):

    import numpy as np 
    sagittal_img = np.sum(pir_vol, axis=2)
    coronal_img = np.sum(pir_vol, axis=0)

    from SpineSeg.utils import globalNormalization
    sagittal_img_norm = globalNormalization(sagittal_img)
    coronal_img_norm = globalNormalization(coronal_img)

    return sagittal_img_norm, coronal_img_norm


def resize_512_sagittal(img):
    # return a list of image crops

    import numpy as np 

    img_crops = []

    img_copy = np.copy(img)

    h, w = img_copy.shape

    if w <= 512:
        img_copy = np.pad(img_copy, [(0, 0), (0, 512-w)], mode='constant', constant_values=img_copy.min())   

        if h <= 512:
            img_copy = np.pad(img_copy, [(0, 512-h), (0, 0)], mode='constant', constant_values=img_copy.min())
        else:
            img_copy = img_copy[0:512, :]

        img_crops.append(img_copy)

    elif w > 512 and w <= 1024:
        img_copy1 = img_copy[:, 0:512]
        img_copy2 = img_copy[:, 512:]

        img_copy2 = np.pad(img_copy2, [(0, 0), (0, 512-img_copy2.shape[1])], mode='constant', constant_values=img_copy2.min()) 

        if h <= 512:
            img_copy1 = np.pad(img_copy1, [(0, 512-h), (0, 0)], mode='constant', constant_values=img_copy1.min())
            img_copy2 = np.pad(img_copy2, [(0, 512-h), (0, 0)], mode='constant', constant_values=img_copy2.min())
        else:
            img_copy1 = img_copy1[0:512, :]
            img_copy2 = img_copy2[0:512, :]

        img_crops.append(img_copy1)
        img_crops.append(img_copy2)

    else:
        img_copy1 = img_copy[:, 0:512]
        img_copy2 = img_copy[:, 512:1024]

        if h <= 512:
            img_copy1 = np.pad(img_copy1, [(0, 512-h), (0, 0)], mode='constant', constant_values=img_copy1.min())
            img_copy2 = np.pad(img_copy2, [(0, 512-h), (0, 0)], mode='constant', constant_values=img_copy2.min())
        else:
            img_copy1 = img_copy1[0:512, :]
            img_copy2 = img_copy2[0:512, :]

        img_crops.append(img_copy1)
        img_crops.append(img_copy2) 


    for img_crop in img_crops:
        assert img_crop.shape == (512, 512)


    return img_crops


def resize_512_coronal(img):
    # return a list of image crops

    import numpy as np 

    img_crops = []

    img_copy = np.copy(img)

    h, w = img_copy.shape

    if h <= 512:
        img_copy = np.pad(img_copy, [(0, 512-h), (0, 0)], mode='constant', constant_values=img_copy.min())   

        if w <= 512:
            img_copy = np.pad(img_copy, [(0, 0), (0, 512-w)], mode='constant', constant_values=img_copy.min())
        else:
            img_copy = img_copy[:, 0:512]

        img_crops.append(img_copy)

    elif h > 512 and h <= 1024:
        img_copy1 = img_copy[0:512, :]
        img_copy2 = img_copy[512:, :]

        img_copy2 = np.pad(img_copy2, [(0, 512-img_copy2.shape[0]), (0, 0)], mode='constant', constant_values=img_copy2.min()) 

        if w <= 512:
            img_copy1 = np.pad(img_copy1, [(0, 0), (0, 512-w)], mode='constant', constant_values=img_copy1.min())
            img_copy2 = np.pad(img_copy2, [(0, 0), (0, 512-w)], mode='constant', constant_values=img_copy2.min())
        else:
            img_copy1 = img_copy1[:, 0:512]
            img_copy2 = img_copy2[:, 0:512]

        img_crops.append(img_copy1)
        img_crops.append(img_copy2)

    else:
        img_copy1 = img_copy[0:512, :]
        img_copy2 = img_copy[512:1024, :]

        if w <= 512:
            img_copy1 = np.pad(img_copy1, [(0, 0), (0, 512-w)], mode='constant', constant_values=img_copy1.min())
            img_copy2 = np.pad(img_copy2, [(0, 0), (0, 512-w)], mode='constant', constant_values=img_copy2.min())
        else:
            img_copy1 = img_copy1[:, 0:512]
            img_copy2 = img_copy2[:, 0:512]

        img_crops.append(img_copy1)
        img_crops.append(img_copy2)        


    for img_crop in img_crops:
        assert img_crop.shape == (512, 512)


    return img_crops


def load_locator_model(model_file):
    import torch 
    from SpineSeg.third_party.locator import unet_model
    model = unet_model.UNet(1, 1,
                            height=512,
                            width=512, 
                            known_n_points=None)

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available(): model = model.cuda()
    model.eval()

    return model 

def toTensor(x):
    import torch

    out = torch.from_numpy(x).to(dtype=torch.float32)
    out = out.view(1, 1, out.size(0), out.size(1))
    if torch.cuda.is_available(): out = out.cuda()

    return out 

def toNumpy(x):
    import numpy as np 

    out = x.view(x.size(-2), x.size(-1))
    out = out.cpu()
    out = out.detach().numpy().astype(np.float32)

    return out 


def model_apply(img, model):

    pred_maps = []

    for im in img:

        im = toTensor(im)
        pred_map, _ = model(im)
        pred_map = toNumpy(pred_map)

        pred_maps.append(pred_map)

    return pred_maps


def resize_back(out, ori_h, ori_w, view):
    # input: a list of probability maps
    # output: a full probability map

    import numpy as np 

    if len(out) == 1:
        pred = out[0]

        h, w = pred.shape

        if h >= ori_h:
            pred = pred[0:ori_h, :]
        else:
            pred = np.pad(pred, [(0, ori_h-h), (0, 0)], mode='constant', constant_values=0)

        if w >= ori_w:
            pred = pred[:, 0:ori_w]
        else:
            pred = np.pad(pred, [(0, 0), (0, ori_w-w)], mode='constant', constant_values=0)

        assert pred.shape == (ori_h, ori_w)

        return pred 

    elif len(out) == 2:

        if view == 'sagittal':

            pred1 = out[0]
            pred2 = out[1]

            h1, w1 = pred1.shape
            h2, w2 = pred2.shape

            assert h1 == h2 and w1 == 512

            if (w1 + w2) >= ori_w:
                pred2 = pred2[:, 0:(ori_w-w1)]
            else:
                pred2 = np.pad(pred2, [(0, 0), (0, ori_w-w1-w2)], mode='constant', constant_values=0)

            if h1 >= ori_h:
                pred1 = pred1[0:ori_h, :]
                pred2 = pred2[0:ori_h, :]
            else:
                pred1 = np.pad(pred1, [(0, ori_h-h1), (0, 0)], mode='constant', constant_values=0)
                pred2 = np.pad(pred2, [(0, ori_h-h2), (0, 0)], mode='constant', constant_values=0)

            assert pred1.shape[0] == pred2.shape[0]
            pred = np.concatenate((pred1, pred2), axis=1)

        elif view == 'coronal':

            pred1 = out[0]
            pred2 = out[1]


            h1, w1 = pred1.shape
            h2, w2 = pred2.shape

            assert w1 == w2 and h1 == 512

            if (h1 + h2) >= ori_h:
                pred2 = pred2[0:(ori_h-h1), :]
            else:
                pred2 = np.pad(pred2, [(0, ori_h-h1-h2), (0, 0)], mode='constant', constant_values=0)

            if w1 >= ori_w:
                pred1 = pred1[:, 0:ori_w]
                pred2 = pred2[:, 0:ori_w]
            else:
                pred1 = np.pad(pred1, [(0, 0), (0, ori_w-w1)], mode='constant', constant_values=0)
                pred2 = np.pad(pred2, [(0, 0), (0, ori_w-w1)], mode='constant', constant_values=0)

            assert pred1.shape[1] == pred2.shape[1]
            pred = np.concatenate((pred1, pred2), axis=0)

        else:
            raise NotImplementedError('unknown plane.')

        assert pred.shape == (ori_h, ori_w)

        return pred

    else:
        raise ValueError('wrong number of probability maps in the input list.')


def get_connected_component(pred):

    import numpy as np 
    from scipy.ndimage.measurements import label

    labeled, ncomponents = label(pred)

    return labeled, ncomponents


def get_loc_from_pred(pred_map):
    import numpy as np 
    labeled, ncomponents = get_connected_component(pred_map)

    loc = []
    for label in range(1, ncomponents+1):

        new_map = np.copy(labeled)
        new_map[new_map!=label] = 0
        x_array, y_array = np.where(new_map>0)

        x_cen = np.mean(x_array).astype(int)
        y_cen = np.mean(y_array).astype(int)

        loc.append([x_cen, y_cen])

    loc = np.array(loc)

    return loc


def locations_from_prediction_map(pred_map, ori_h, ori_w, view):

    pred_map = resize_back(pred_map, ori_h, ori_w, view)

    from skimage import filters
    th = filters.threshold_otsu(pred_map)
    binary_out = (pred_map>th).astype(int)

    loc = get_loc_from_pred(binary_out)

    return loc 


def get_left_neighboor_idx(idx, zeros_list):

    left_idx = idx - 1
    while left_idx in zeros_list:
        left_idx -= 1

    return left_idx


def get_right_neighboor_idx(idx, zeros_list):

    right_idx = idx + 1
    while right_idx in zeros_list:
        right_idx += 1

    return right_idx


def refine_z(coordinate_z):

    import numpy as np 
    import warnings

    if (coordinate_z==0).all():
        warnings.warn('The z coordinates are empty for all the detected locations.')

        return coordinate_z

    idx_zeros = np.where(coordinate_z==0)[0]

    for idx in idx_zeros:
        left_idx = get_left_neighboor_idx(idx, idx_zeros)
        right_idx = get_right_neighboor_idx(idx, idx_zeros)

        if left_idx < 0:
            coordinate_z[idx] = coordinate_z[right_idx]
        elif right_idx >= len(coordinate_z):
            coordinate_z[idx] = coordinate_z[left_idx]
        else:
            coordinate_z[idx] = (coordinate_z[left_idx] + coordinate_z[right_idx])//2

    return coordinate_z


def get_nearest_neighboor_idx(y, vector_y):

    import numpy as np 
    return np.argmin(abs(vector_y-y))


def aggreagte_biplanar_locations(loc_pred_sagittal, loc_pred_coronal, radius=10):
    import numpy as np 

    y_sagittal = loc_pred_sagittal[:,1]
    y_coronal = loc_pred_coronal[:,0]

    coordinate = np.zeros((loc_pred_sagittal.shape[0], loc_pred_sagittal.shape[1]+1))
    coordinate[:,0:2] = loc_pred_sagittal

    matching_idx = []
    
    for i, y_s in enumerate(y_sagittal):
        j = get_nearest_neighboor_idx(y_s, y_coronal)
        if abs(y_s-y_coronal[j]) < radius:
            matching_idx.append([i,j])


    for i,j in matching_idx:
        coordinate[i,2] = loc_pred_coronal[j,1]

    coordinate_z = coordinate[:,2]
    if np.count_nonzero(coordinate_z==0) > 0:
        coordinate_z = refine_z(coordinate_z)
        coordinate[:,2] = coordinate_z

    return coordinate


def locate(pir_vol, sagittal_model_file, coronal_model_file):
    import numpy as np
    sagittal_img_norm, coronal_img_norm = get_norm_sagittal_coronal_images(pir_vol)

    # np.save("./sample/seg_img", sagittal_img_norm)
    # np.save("./sample/cor_img", coronal_img_norm)


    sagittal_model = load_locator_model(sagittal_model_file)
    coronal_model = load_locator_model(coronal_model_file)

    sagittal_img_512 = resize_512_sagittal(sagittal_img_norm)
    coronal_img_512 = resize_512_coronal(coronal_img_norm)

    sagittal_pred_map = model_apply(sagittal_img_512, sagittal_model)
    coronal_pred_map = model_apply(coronal_img_512, coronal_model)

    loc_sag = locations_from_prediction_map(sagittal_pred_map, sagittal_img_norm.shape[0], sagittal_img_norm.shape[1], 'sagittal')
    loc_cor = locations_from_prediction_map(coronal_pred_map, coronal_img_norm.shape[0], coronal_img_norm.shape[1], 'coronal')

    coordinate = aggreagte_biplanar_locations(loc_sag, loc_cor)

    coordinate = coordinate[coordinate[:,1].argsort()]

    print("\npred loc:",coordinate)

    save_sag_img_file = "/media/alg/data3/DeepSpineData/spine_test/Test09/predict_spine/sag.png"
    draw_points2png(sagittal_img_norm, loc_sag, save_sag_img_file)

    save_cor_img_file = "/media/alg/data3/DeepSpineData/spine_test/Test09/predict_spine/cor.png"
    draw_points2png(coronal_img_norm, loc_cor, save_cor_img_file)


    return coordinate


def draw_points2png(data, locations, save_img_file):
    import numpy as np
    from PIL import Image, ImageDraw
    data_min = np.min(data)
    data_max = np.max(data)
    data = (data - data_min) / (data_max - data_min) * 255

    img = Image.fromarray(data.astype('uint8'))
    img = img.convert("RGB")
    drawing = ImageDraw.Draw(img)
    for loc in locations:
        y = loc[0]
        x = loc[1]
        drawing.ellipse(((x-3, y-3), (x+3, y+3)), fill=(0, 0,255))
    img.save(save_img_file)







# def convert_drr_txt2png(drr_txt_file, save_png_file):
#     """
#
#     :param dtt_txt_file:
#     :param save_png_file:
#     :return:
#     """
#     data = np.loadtxt(drr_txt_file)
#     data=data[::-1, :]
#     data_min = np.min(data)
#     data_max = np.max(data)
#     data = (data-data_min)/(data_max-data_min)*255
#
#     image_out = Image.fromarray(data.astype('uint8'))
#     img_as_img = image_out.convert("RGB")
#     img_as_img.save(save_png_file)
#     # ans = plt.imshow(data, cmap=plt.cm.gray)
#     # plt.colorbar()
#     # plt.show()
#     print("down")
