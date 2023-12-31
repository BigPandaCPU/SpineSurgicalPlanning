__copyright__ = \
"""
Copyright &copyright © (c) 2021 Inria Grenoble Rhône-Alpes.
All rights reserved.

This source code is to be used for academic research purposes only.
For commercial uses of the code, please send an email to edmond.boyer@inria.fr and sergi.pujades@inria.fr

"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Di Meng"


import time
from PySiddonGPU.gernerateDRR import generateDRR
from yolov5.detect import yolo_predict


def load_models(seg_spine_norm=False, seg_vert_norm=False):

    if seg_spine_norm:
        model_file_seg_binary = '../models/segmentor_spine_norm.pth'
    else:
        model_file_seg_binary = '../models/segmentor_spine.pth'

    if seg_vert_norm:
        model_file_seg_idv = '../models/segmentor_vertebra_norm.pth'
    else:
        model_file_seg_idv = '../models/segmentor_vertebra.pth'

    model_file_loc_sag = '../models/locator_sagittal.pth'
    model_file_loc_cor = '../models/locator_coronal.pth'

    id_group_model_file = '../models/classifier_group.pth'
    id_cer_model_file = '../models/classifier_cervical.pth'
    id_thor_model_file = '../models/classifier_thoracic.pth'
    id_lum_model_file = '../models/classifier_lumbar.pth'


    return {'seg_binary': model_file_seg_binary, 'seg_individual': model_file_seg_idv,
            'loc_sagittal': model_file_loc_sag, 'loc_coronal': model_file_loc_cor,
            'id_group': id_group_model_file, 'id_cervical': id_cer_model_file,
            'id_thoracic': id_thor_model_file, 'id_lumbar': id_lum_model_file}

def loadTxt(txt_file):
    fp = open(txt_file)
    lines = fp.readlines()
    fp.close()
    line_values = []
    for line in lines:
        line = line.strip()
        line_list = line.split(" ")
        cur_line_value = []
        for i in range(1, len(line_list)):
            line_value = line_list[i]
            cur_line_value.append(float(line_value))
        line_values.append(cur_line_value)
    return line_values

def convertNormalBox2RealBox(boxs, img_w, img_h):
    boxs_xyxy = []
    for box in boxs:
        box_xyxy = [None]*4
        box_xyxy[0] = int((box[0] - box[2]/2.0)*img_w)
        box_xyxy[1] = int((box[1] - box[3]/2.0)*img_h)
        box_xyxy[2] = int((box[0] + box[2]/2.0)*img_w)
        box_xyxy[3] = int((box[1] + box[3]/2.0)*img_h)
        boxs_xyxy.append(box_xyxy)
    return boxs_xyxy

def getCropBox(drr_output_folder, yolo_output_folder):
    img_info_file = os.path.join(drr_output_folder, "img_info.txt")
    imgX, imgY, imgZ = np.loadtxt(img_info_file).astype(int)


    drr_front_txt_file = os.path.join(yolo_output_folder, 'DRR_front.txt')
    if not os.path.exists(drr_front_txt_file):
        return None
    drr_front_boxs_xywh = loadTxt(drr_front_txt_file)
    drr_front_boxs_xyxy = convertNormalBox2RealBox(drr_front_boxs_xywh, imgX, imgZ)

    #assert len(drr_front_boxs_xyxy) == 1, "there are more than one hip "
    if len(drr_front_boxs_xyxy) > 1:
        return None

    #print(drr_front_boxs_xyxy)

    drr_side_txt_file = os.path.join(yolo_output_folder, "DRR_side.txt")
    if not os.path.exists(drr_side_txt_file):
        return None

    drr_side_boxs_xywh = loadTxt(drr_side_txt_file)
    drr_side_boxs_xyxy = convertNormalBox2RealBox(drr_side_boxs_xywh, imgY, imgZ)

    if len(drr_side_boxs_xyxy) > 1:
        drr_side_box_xyxy = getAimBox(drr_side_boxs_xyxy, drr_front_boxs_xyxy[0])
    else:
        drr_side_box_xyxy = drr_side_boxs_xyxy[0]
    drr_front_box_xyxy = drr_front_boxs_xyxy[0]
    crop_box_mixX = np.max([drr_front_box_xyxy[0]-10, 0])
    crop_box_maxX = np.min([drr_front_box_xyxy[2]+10, imgX])
    crop_box_mixY = np.max([drr_side_box_xyxy[0]-10, 0])
    crop_box_maxY = np.min([drr_side_box_xyxy[2]+10, imgY])
    crop_box_mixZ = int(np.min([drr_front_box_xyxy[1], drr_side_box_xyxy[1]]))
    crop_box_maxZ = int(np.max([drr_front_box_xyxy[3], drr_side_box_xyxy[3]]))

    crop_boxs = [[crop_box_mixX, crop_box_maxX, crop_box_mixY, crop_box_maxY, imgZ - crop_box_maxZ, imgZ-crop_box_mixZ]]
    return crop_boxs

def getAimBox(source_boxs, reference_box):
    for source_box in source_boxs:
        stack_minY = np.max([source_box[1], reference_box[1]])
        stack_maxY = np.min([source_box[3], reference_box[3]])

        if stack_maxY > stack_minY:
            return source_box

def removeAllFiles(src_dir):
    file_names = os.listdir(src_dir)
    for file_name in file_names:
        cur_file = os.path.join(src_dir, file_name)
        if os.path.isfile(cur_file):
            os.remove(cur_file)



if __name__ == "__main__":

    import argparse, os, glob, sys
    from utils import mkpath, read_isotropic_pir_img_from_nifti_file, read_isotropic_pir_img_from_nifti_img,write_result_to_file,get_result_of_mask
    import numpy as np
    import torch
    import time
    from tqdm import tqdm
    import nibabel as nib
    torch.set_grad_enabled(False)

    import SimpleITK as sitk


    parser = argparse.ArgumentParser(description='Run pipeline on a single CT scan.')
    parser.add_argument('-D', '--input_data', type=str, help='a CT scan or a folder of CT scans in nifti format')
    parser.add_argument('-S', '--save_folder',  default='-1', type=str, help='folder to save the results')
    parser.add_argument('-F', '--force_recompute', action='store_true', help='set True to recompute and overwrite the results')
    parser.add_argument('-L', '--initial_locations', action='store_true', help='set True to use initial location predictions')
    parser.add_argument('-Ns', '--seg_spine_norm', action='store_true', help='set True to use normalized spine segmentor')
    parser.add_argument('-Nv', '--seg_vert_norm', action='store_true', help='set True to use normalized vertebra segmentor')
    args = parser.parse_args()


    src_dir = "/media/alg/data3/DeepSpineData/CTSpine1k_new/image/good/"
    save_dir = "/media/alg/data3/DeepSpineData/CTSpine1k_new/image/predict_verSeg"

    img_names = os.listdir(src_dir)
    for img_name in tqdm(img_names):
        time_sum_start = time.time()
        shot_name = img_name[:-7]
        save_folder = os.path.join(save_dir, shot_name)
        #
        if "Test" in img_name:
            continue
        # if os.path.exists(save_folder):
        #     continue

        ### results saving locations
        #save_folder = args.save_folder
        if save_folder != '-1':
            mkpath(save_folder)
        else:
            current_path = os.path.abspath(os.getcwd())
            save_folder = os.path.join(current_path, 'results')
            mkpath(save_folder)


        scanname = shot_name
        scan_file = os.path.join(src_dir, img_name)


        print(' ... starting to process: ', scan_file)

        # =================================================================
        # DRR module
        # =================================================================

        #  *** step0:preprare dir ***
        output_folder = save_folder
        drr_output_folder = os.path.join(output_folder, "drr")
        yolo_output_folder = os.path.join(output_folder, 'yolo')


        mode = "Spine"

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(drr_output_folder, exist_ok=True)
        os.makedirs(yolo_output_folder, exist_ok=True)
        removeAllFiles(drr_output_folder)
        removeAllFiles(yolo_output_folder)

        #  *** step1: generate DRR img ***
        input_file = scan_file
        drr_ouput_folder = os.path.join(output_folder, "drr")
        img_data, properties = generateDRR(input_file, drr_output_folder)

        #  *** step2: bbox detection from yolov5 ***  #
        yolo_predict(output_folder, mode)

        time_yolo = time.time()

        # *** step3: load bbox  *** #
        crop_boxs = getCropBox(drr_output_folder, yolo_output_folder)
        time_boxs = time.time()
        crop_box = crop_boxs[0]
        x_min, x_max, y_min, y_max, z_min, z_max = crop_box

        # *** step4: aim img data and parameters prepare *** #
        spacing = properties['itk_spacing']
        origin = properties['itk_origin']
        direction = properties['itk_direction']

        img_data_trans = img_data.transpose(2,1,0)

        affine = np.zeros([4,4])
        affine[0, 0] = -spacing[0]
        affine[1, 1] = -spacing[1]
        affine[2, 2] = spacing[2]
        affine[0:3,3] = origin*np.array([-1, -1, 1])
        affine[3,3] = 1.0

        ori_size = img_data_trans.shape
        ori_spacing = np.array(spacing)
        ori_aff = affine
        a, b, c = nib.orientations.aff2axcodes(ori_aff)
        ori_orient_code = a + b + c


        nifti_img = nib.Nifti1Image(img_data_trans, ori_aff)

        # =================================================================
        # Load the CT scan
        # =================================================================

        ### TODO: data I/O for other formats

        time_load_start = time.time()
        try:
            #pir_img_all = read_isotropic_pir_img_from_nifti_file(scan_file) #(y,z,x)
            #pir_img_all = read_isotropic_pir_img_from_img_data(img_data_trans, ori_spacing, ori_orient_code, itm_orient="PIR")
            pir_img_all = read_isotropic_pir_img_from_nifti_img(nifti_img, ori_spacing, ori_orient_code)
        except ImageFileError:
            sys.exit('The input CT should be in nifti format.')

        y_min_new = int(y_min*spacing[1])
        y_max_new = int(y_max*spacing[1])
        z_min_new = int(z_min*spacing[2])
        z_max_new = int(z_max*spacing[2])
        x_min_new = int(x_min*spacing[0])
        x_max_new = int(x_max*spacing[0])

        pir_img_all_Z, pir_img_all_Y, pir_img_all_X = pir_img_all.shape

        pir_img = pir_img_all[y_min_new:y_max_new, pir_img_all_Y-z_max_new:pir_img_all_Y-z_min_new, x_min_new:x_max_new]
        # pir_img_itk = sitk.GetImageFromArray(pir_img)
        # sitk.WriteImage(pir_img_itk, os.path.join(output_folder, "pir_img.nii.gz"))
        print(' ... loaded CT volume in isotropic resolution and PIR orientation ')
        time_load_end = time.time()

        # =================================================================
        # Spine binary segmentation
        # =================================================================

        models = load_models(seg_spine_norm=False, seg_vert_norm=False)

        time_binary_start = time.time()
        from segment_spine import binary_segmentor
        binary_mask = binary_segmentor(pir_img, models['seg_binary'], mode='overlap', norm=args.seg_spine_norm)
        #np.save("./sample/binary_mask", binary_mask)
        #binary_mask = np.load("./sample/Test10/Test10_binary_mask.npy")

        # mask_itk = sitk.GetImageFromArray(binary_mask)
        # save_mask_file = os.path.join(output_folder, "crop_seg.nii.gz")
        # mask_itk.SetSpacing(spacing)
        # mask_itk.SetOrigin(origin)
        # mask_itk.SetDirection(direction)
        # sitk.WriteImage(mask_itk, save_mask_file)
        time_binary_end = time.time()

        print(' ... obtained spine binary segmentation ')


        # =================================================================
        # Initial locations
        # =================================================================

        time_locate_start = time.time()
        locations = np.array([])
        #if args.initial_locations:
        from locate import locate
        locations = locate(pir_img, models['loc_sagittal'], models['loc_coronal'])
        time_locate_end = time.time()
        print(' ... obtained {} initial 3D locations '.format(len(locations)))

        # =================================================================
        # Consistency circle - Locations refine - multi label segmentation
        # =================================================================

        time_con_start = time.time()
        from consistency_loop import consistency_refinement_close_loop

        multi_label_mask, locations, labels, loc_has_converged = consistency_refinement_close_loop(locations, pir_img, binary_mask,
                                                                            models['seg_individual'], args.seg_vert_norm,
                                                                            models['id_group'], models['id_cervical'],
                                                                            models['id_thoracic'], models['id_lumbar'])

        np.save("./sample/loc", locations)
        print("\nfinall loc:", locations)

        time_con_end = time.time()
        print(' ... obtained PIR multi label segmentation ')

        # =================================================================
        # Save the result in original format
        # =================================================================
        time_save_start = time.time()
        # from utils import get_size_and_spacing_and_orientation_from_nifti_file, write_result_to_file
        # ori_size, ori_spacing, ori_orient_code, ori_aff = get_size_and_spacing_and_orientation_from_nifti_file(scan_file)

        mask = np.zeros_like(pir_img_all)
        mask[y_min_new:y_max_new, pir_img_all_Y-z_max_new:pir_img_all_Y-z_min_new, x_min_new:x_max_new] = multi_label_mask

        #mask = mask.transpose(2,1,0)
        # mask_itk = sitk.GetImageFromArray(mask)
        # save_itk_file = os.path.join(save_folder, "mask.nii.gz")
        # sitk.WriteImage(mask_itk, save_itk_file)


        write_result_to_file(mask, ori_orient_code, ori_spacing, ori_size, ori_aff, save_folder, scanname)
        #crop_mask = get_result_of_mask(multi_label_mask, ori_orient_code, ori_spacing, ori_size, ori_aff, save_folder, scanname)

        # mask_np = np.zeros(img_data.shape)
        # mask_np[z_min:z_max, y_min:y_max, x_min:x_max] = crop_mask
        # mask_itk = sitk.GetImageFromArray(mask_np)
        # mask_itk.SetDirection(direction)
        # mask_itk.SetOrigin(origin)
        # mask_itk.SetDirection(direction)
        # save_mask_file = os.path.join(save_folder, scanname+"_seg.nii.gz")
        # sitk.WriteImage(mask_itk, save_mask_file)
        #
        #
        time_end = time.time()

        time_save_end = time.time()
        time_sum_end = time.time()
        print("\n\nload img time:", time_load_end - time_load_start)
        print("locate time:", time_locate_end - time_locate_start)
        print("binary seg time:", time_binary_end - time_binary_start)
        print("con time:", time_con_end - time_con_start)
        print("save time:", time_save_end - time_save_start)
        print("sum time:", time_sum_end - time_sum_start)
        print("\n\n\n")