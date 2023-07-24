
def get_cubes(img, msk, cube_size, stride, norm=False):
    import numpy as np 
    from utils import globalNormalization

    if min(img.shape) < cube_size:
        img = vol_padding(img, cube_size, pad_value=-1000)
        msk = vol_padding(msk, cube_size, pad_value=0)

    if norm:
        img = globalNormalization(img)

    print('labels in the mask: ', np.unique(msk))

    h, w, c = img.shape

    img_cubes = []
    msk_cubes = []


    for i in range(0, h-cube_size+1, stride):
        for j in range(0, w-cube_size+1, stride):
            for k in range(0, c-cube_size+1, stride):

                img_cube = img[i:i+cube_size, j:j+cube_size, k:k+cube_size]
                msk_cube = msk[i:i+cube_size, j:j+cube_size, k:k+cube_size]

                img_cubes.append(img_cube)
                msk_cubes.append(msk_cube)


    img_cubes = np.array(img_cubes, dtype=np.float32)
    msk_cubes = np.array(msk_cubes, dtype=np.float32)

    return img_cubes, msk_cubes


def vol_padding(vol, cube_size, pad_value=-1000):   # pad CT with -1000 (HU of air)
    import numpy as np 
    h, w, c = vol.shape

    pad_value = float(pad_value)

    if h < cube_size:
        pad_width_1 = (cube_size - h)//2
        pad_width_2 = cube_size - h - pad_width_1
        vol = np.pad(vol, [(pad_width_1, pad_width_2), (0, 0), (0, 0)], mode='constant', constant_values=pad_value)

    if w < cube_size:
        pad_width_1 = (cube_size - w)//2
        pad_width_2 = cube_size - w - pad_width_1
        vol = np.pad(vol, [(0, 0), (pad_width_1, pad_width_2), (0, 0)], mode='constant', constant_values=pad_value)

    if c < cube_size:
        pad_width_1 = (cube_size - c)//2
        pad_width_2 = cube_size - c - pad_width_1
        vol = np.pad(vol, [(0, 0), (0, 0), (pad_width_1, pad_width_2)], mode='constant', constant_values=pad_value)

    assert min(vol.shape) >= cube_size

    return vol


def get_roi_cubes(img, msk, cube_size, stride, norm=False):
    import numpy as np 
    from utils import globalNormalization

    h, w, c = img.shape

    min_x, min_y, min_z, max_x, max_y, max_z = get_roi_bbox(msk)

    roi_h = max_x - min_x
    roi_w = max_y - min_y
    roi_c = max_z - min_z

    if not roi_h >= cube_size and roi_w >= cube_size and roi_c >= cube_size:
        min_x, min_y, min_z, max_x, max_y, max_z = positioning_roi(min_x, min_y, min_z, 
                                                                   max_x, max_y, max_z, 
                                                                   h, w, c, cube_size)

    img_roi = img[min_x:max_x, min_y:max_y, min_z:max_z]
    msk_roi = msk[min_x:max_x, min_y:max_y, min_z:max_z]

    if min(img_roi.shape) < cube_size:
        img_roi = vol_padding(img_roi, cube_size, pad_value=-1000)
        msk_roi = vol_padding(msk_roi, cube_size, pad_value=0)

    if norm:
        img_roi = globalNormalization(img_roi)

    h, w, c = img_roi.shape

    img_cubes = []
    msk_cubes = []

    for i in range(0, h-cube_size+1, stride):
        for j in range(0, w-cube_size+1, stride):
            for k in range(0, c-cube_size+1, stride):

                img_cube = img_roi[i:i+cube_size, j:j+cube_size, k:k+cube_size]
                msk_cube = msk_roi[i:i+cube_size, j:j+cube_size, k:k+cube_size]

                img_cubes.append(img_cube)
                msk_cubes.append(msk_cube)

    img_cubes = np.array(img_cubes, dtype=np.float32)
    msk_cubes = np.array(msk_cubes, dtype=np.float32)

    return img_cubes, msk_cubes


def get_roi_bbox(msk):
    import numpy as np 

    x_array, y_array, z_array = np.where(msk>0)

    min_x, max_x = min(x_array), max(x_array)
    min_y, max_y = min(y_array), max(y_array)
    min_z, max_z = min(z_array), max(z_array)

    return min_x, min_y, min_z, max_x, max_y, max_z


def positioning_roi(min_x, min_y, min_z, max_x, max_y, max_z, h, w, c, cube_size):

    roi_h = max_x - min_x
    roi_w = max_y - min_y
    roi_c = max_z - min_z


    min_x, max_x = positioning_roi_width(min_x, max_x, roi_h, cube_size, h)
    min_y, max_y = positioning_roi_width(min_y, max_y, roi_w, cube_size, w)
    min_z, max_z = positioning_roi_width(min_z, max_z, roi_c, cube_size, c)

    return min_x, min_y, min_z, max_x, max_y, max_z


def positioning_roi_width(min_x, max_x, roi_h, cube_size, h):

    if roi_h < cube_size:
        ready_flag = True
        expand_width = cube_size - roi_h
        if min_x - expand_width//2 >= 0:
            min_x =- expand_width//2 
        else:
            min_x = 0
            ready_flag = False
        if max_x + (expand_width - expand_width//2) <= h and ready_flag:
            max_x += (expand_width - expand_width//2) 

        elif not ready_flag:
            if cube_size <=h:
                max_x = cube_size
        elif max_x + (expand_width - expand_width//2) > h and ready_flag:
            max_x = h
        else:
            pass

    return min_x, max_x


def ROI_pad(anchor, vol_size, ROI_size):  

    h, w, c = vol_size[:]
    x, y, z = anchor[:]
    rx, ry, rz = ROI_size[:]

    padX = [0, 0]
    padY = [0, 0]
    padZ = [0, 0]

    if x - rx//2 < 0:
        padX[0] = rx//2 - x
    if x + rx//2 > h:
        padX[1] = x + rx//2 - h 

    if y - ry//2 < 0:
        padY[0] = ry//2 - y 
    if y + ry//2 > w:
        padY[1] = y + ry//2 - w 

    if z - rz//2 < 0:
        padZ[0] = rz//2 - z 
    if z + rz//2 > c:
        padZ[1] = z + rz//2 - c 

    return padX, padY, padZ


def get_ROI_img(vol, anchor):
    import numpy as np 
    from scipy import ndimage

    anchor = np.round(anchor).astype(np.int)

    x, y, z = anchor[:]

    ROI_size = (200, 200, 200) 

    h, w, c = vol.shape

    padX, padY, padZ = ROI_pad(anchor, vol.shape, ROI_size)

    img_crop = vol[max(x-100, 0):min(x+100, h), max(y-100, 0):min(y+100, w), max(z-100, 0):min(z+100, c)]

    img_pad = np.pad(img_crop, [padX, padY, padZ], mode='constant', constant_values=vol.min())

    assert img_pad.shape == ROI_size

    return img_pad


def rotate_img(img, angle):

    from scipy import ndimage

    imgR = ndimage.rotate(img, angle, cval=img.min(), reshape=False)

    return imgR


def read_image_and_mask(ID, dataset_dir, pir_orientation):
    import glob, os
    from utils import read_nifti_file, read_isotropic_pir_img_from_nifti_file

    # read the image and mask 
    vol_files = glob.glob(os.path.join(dataset_dir, ID, '{}_*.nii.gz').format(ID))

    for file in vol_files:
        if '_seg' in file:
            msk_file = file 
        else:
            img_file = file 

    if pir_orientation:
        img = read_isotropic_pir_img_from_nifti_file(img_file)
        msk = read_isotropic_pir_img_from_nifti_file(msk_file)
    else:
        img = read_nifti_file(img_file)
        msk = read_nifti_file(msk_file)

    return img, msk

def read_annotation(ID, dataset_dir):
    import glob, os
    from utils import read_json_file
    import numpy as np 

    # read the labels and locations
    anno_file = glob.glob(os.path.join(dataset_dir, ID, '{}_*.json').format(ID))[0]
    anno = read_json_file(anno_file)

    locs = []
    labels = []

    for i in range(len(anno)):
        x = int(anno[i]['X'])
        y = int(anno[i]['Y'])
        z = int(anno[i]['Z'])
        label = int(anno[i]['label'])

        locs.append([x,y,z])
        labels.append(label)

    locs = np.array(locs).astype(np.float)
    labels = np.array(labels)

    annotations = {'locations': locs, 'labels': labels}

    return annotations


def generate_idv_segmentor_paired_cubes_per_ID(ID, dataset_dir, save_folder):

    import numpy as np 
    import os 
    from scipy import ndimage
    import random 
    import math
    from utils import save_to_nifti_file


    img_save_dir = os.path.join(save_folder, 'img')
    msk_save_dir = os.path.join(save_folder, 'msk')

    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    if not os.path.exists(msk_save_dir):
        os.mkdir(msk_save_dir)


    print(' ... processing ID: ', ID)

    pir_img, pir_msk = read_image_and_mask(ID, dataset_dir, pir_orientation=True)

    anno = read_annotation(ID, dataset_dir)
    locations = anno['locations']
    labels = anno['labels']       


    for i, (loc, label) in enumerate(zip(locations, labels)):

        if i != len(labels) - 1:
            next_loc = locations[i+1]
            angle_to_vertical = np.arctan((next_loc[0]-loc[0])/(next_loc[1]-loc[1])) * 180 / np.pi
        else:
            pre_loc = locations[i-1]
            # print('using previous location to compute angle')
            angle_to_vertical = np.arctan((loc[0]-pre_loc[0])/(loc[1]-pre_loc[1])) * 180 / np.pi     

        # print('angle to vertical: ', angle_to_vertical)

        gt_msk = (pir_msk==label).astype(np.float)
        gt_img = np.copy(pir_img)


        img_roi = get_ROI_img(gt_img, loc) 
        msk_roi = get_ROI_img(gt_msk, loc)
        msk_roi = np.round(msk_roi).astype(np.int)

        assert len(np.unique(msk_roi)) == 2
        assert img_roi.shape == msk_roi.shape 


        # augmented rotation range
        min_rotz = int(min(-50, angle_to_vertical-50))
        max_rotz = int(max(50, angle_to_vertical+50))

        # print('min rotz: {}, max rotz: {}'.format(min_rotz, max_rotz))

        # sampling extra 5 augmented data
        rot_step = math.ceil((max_rotz - min_rotz)/5)

        rot_range = [] 
        for rot in range(max_rotz, min_rotz, -rot_step):
            rot_range.append(rot)

        rot_range.append(0)


        for rot_z in rot_range:

            img_rot = rotate_img(img_roi, rot_z) 
            msk_rot = rotate_img(msk_roi, rot_z)
            msk_rot = np.round(msk_rot).astype(np.int)


            for shift_x in [-10, 0, 10]:
                for shift_y in [-10, 0, 10]:
                    for shift_z in [-10, 0, 10]:     

                        img_rot_copy = np.copy(img_rot)
                        msk_rot_copy = np.copy(msk_rot)

                        if shift_x < 0:
                            size_range_x = [10, 200]
                        else:
                            size_range_x = [0, 190]

                        if shift_y < 0:
                            size_range_y = [10, 200]
                        else:
                            size_range_y = [0, 190]

                        if shift_z < 0:
                            size_range_z = [10, 200]
                        else:
                            size_range_z = [0, 190]

                        img_sroi = img_rot_copy[size_range_x[0]:size_range_x[1], size_range_y[0]:size_range_y[1],
                                                size_range_z[0]:size_range_z[1]]
                        msk_sroi = msk_rot_copy[size_range_x[0]:size_range_x[1], size_range_y[0]:size_range_y[1],
                                                size_range_z[0]:size_range_z[1]]  


                        img_cube = img_sroi[31+20:159+20, 31:159, 31:159]
                        msk_cube = msk_sroi[31+20:159+20, 31:159, 31:159]

                        save_filename = '{}_bone{}_shift_{}_{}_{}_rotz{}.nii.gz'.format(ID, str(label), 
                                                                                 str(shift_x), str(shift_y), str(shift_z),
                                                                                 str(rot_z))


                        save_to_nifti_file(img_cube, os.path.join(img_save_dir, save_filename))
                        save_to_nifti_file(msk_cube, os.path.join(msk_save_dir, save_filename))


def generate_classifier_msk_cubes_with_neighbors_per_ID(ID, dataset_dir, save_dir):
    import os, math 
    import numpy as np 
    from scipy import ndimage
    from utils import save_to_nifti_file


    print(' ... processing ID: ', ID)


    _, pir_msk = read_image_and_mask(ID, dataset_dir, pir_orientation=True)

    anno = read_annotation(ID, dataset_dir)
    locations = anno['locations']
    labels = anno['labels']       
   

    pir_msk = (pir_msk>0).astype(np.int)

    assert len(np.unique(pir_msk)) == 2 

    for i, (loc, label) in enumerate(zip(locations, labels)):


        if i != len(labels) - 1:
            next_loc = locations[i+1]
            angle_to_vertical = np.arctan((next_loc[0]-loc[0])/(next_loc[1]-loc[1])) * 180 / np.pi
        else:
            pre_loc = locations[i-1]
            # print('using previous location to compute angle')
            angle_to_vertical = np.arctan((loc[0]-pre_loc[0])/(loc[1]-pre_loc[1])) * 180 / np.pi     


        msk_roi = get_ROI_img(pir_msk, loc)
        msk_roi = np.round(msk_roi).astype(np.int)

        assert len(np.unique(msk_roi)) == 2

        # rotation range
        min_rotz = int(min(-50, angle_to_vertical-50))
        max_rotz = int(max(50, angle_to_vertical+50))

        # print('min rotz: {}, max rotz: {}'.format(min_rotz, max_rotz))

        # 5 samples in the rotation range
        rot_step = math.ceil((max_rotz - min_rotz)/4)

        rot_range = [] 
        for rot in range(max_rotz, min_rotz, -rot_step):
            rot_range.append(rot)

        # assert len(rot_range) == 5
        rot_range.append(0)


        for rot_z in rot_range:

            msk_rot = rotate_img(msk_roi, rot_z) 
            msk_rot = np.round(msk_rot).astype(np.int)

            for shift_x in [-10, 0, 10]:
                for shift_y in [-10, 0, 10]:
                    for shift_z in [-10, 0, 10]:     

                        msk_rot_copy = np.copy(msk_rot)

                        if shift_x < 0:
                            size_range_x = [10, 200]
                        else:
                            size_range_x = [0, 190]

                        if shift_y < 0:
                            size_range_y = [10, 200]
                        else:
                            size_range_y = [0, 190]

                        if shift_z < 0:
                            size_range_z = [10, 200]
                        else:
                            size_range_z = [0, 190]

                        msk_sroi = msk_rot_copy[size_range_x[0]:size_range_x[1], size_range_y[0]:size_range_y[1],
                                                size_range_z[0]:size_range_z[1]]

                        msk_cube = msk_sroi[31:159, 31:159, 31:159]

                        save_filename = '{}_bone{}_shift_{}_{}_{}_rotz{}.nii.gz'.format(ID, str(label), 
                                                                                 str(shift_x), str(shift_y), str(shift_z),
                                                                                 str(rot_z))


                        save_to_nifti_file(msk_cube, os.path.join(save_dir, save_filename))


def generate_spine_segmentor_paired_cubes_per_ID(ID, dataset_dir, save_dir):
    
    from utils import save_to_nifti_file
    import os
    from scipy import ndimage
    import numpy as np 

    cube_size = 96
    stride = 90
    stride_roi = 30

    img_save_dir = os.path.join(save_folder, 'img')
    msk_save_dir = os.path.join(save_folder, 'msk')

    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    if not os.path.exists(msk_save_dir):
        os.mkdir(msk_save_dir)


    print(' ... processing ID: ', ID)


    pir_img, pir_msk = read_image_and_mask(ID, dataset_dir, pir_orientation=True)

    pir_msk[pir_msk>0] = 1

    count = 0
    for rot_z in range(-30, 30, 10):

        img_rot = ndimage.rotate(pir_img, rot_z, cval=pir_img_copy.min(), reshape=False) 

        msk_rot = ndimage.rotate(pir_msk, rot_z, cval=0, reshape=False)
        msk_rot = np.round(msk_rot)

        img_cubes, msk_cubes = get_cubes(img_rot, msk_rot, cube_size, stride, norm=False)
        img_roi_cubes, msk_roi_cubes = get_roi_cubes(img_rot, msk_rot, cube_size, stride_roi, norm=False)


        img_cubes = np.concatenate((img_cubes, img_roi_cubes), axis=0)
        msk_cubes = np.concatenate((msk_cubes, msk_roi_cubes), axis=0)

        assert img_cubes.shape == msk_cubes.shape
        
        for idx, img_cube in enumerate(img_cubes):

            msk_cube = msk_cubes[idx]

            assert img_cube.shape == msk_cube.shape

            assert msk_cube.max() <= 1

            save_filename_img = os.path.join(save_img_dir, '{}_{:04d}.nii.gz'.format(ID, count))
            save_filename_msk = os.path.join(save_msk_dir, '{}_{:04d}.nii.gz'.format(ID, count))

            save_to_nifti_file(img_cube, save_filename_img)
            save_to_nifti_file(msk_cube, save_filename_msk)

            count += 1


if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description='generate 3D augmented training cubes for segmentors and classifier.')
    parser.add_argument('-T', '--task', type=str, help='options: idv_segmentor | classifier | spine_segmentor')
    parser.add_argument('-D', '--dataset_dir', type=str, help='path to the verse20 training set')
    parser.add_argument('-L', '--ID_list', nargs='+', help='a list of scan IDs, eg. verse008, GL003, ...')
    parser.add_argument('-S', '--save_dir', type=str, help='folder to save the generated 3D cubes')

    args = parser.parse_args()


    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)


    for ID in args.ID_list:

        save_id_dir = os.path.join(args.save_dir, ID)
        if not os.path.exists(save_id_dir):
            os.mkdir(save_id_dir)

        if args.task == 'idv_segmentor':
            generate_idv_segmentor_paired_cubes_per_ID(ID, args.dataset_dir, save_id_dir)
        elif args.task == 'classifier':
            generate_classifier_msk_cubes_with_neighbors_per_ID(ID, args.dataset_dir, save_id_dir)
        elif args.task == 'spine_segmentor':
            generate_spine_segmentor_paired_cubes_per_ID(ID, args.dataset_dir, save_id_dir)
        else:
            raise NotImplementError('Unknown task {}, available options: idv_segmentor | classifier | spine_segmentor'.format(args.task))


