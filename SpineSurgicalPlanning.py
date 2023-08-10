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
import argparse, os, glob, sys
from SpineSeg.utils import mkpath, read_isotropic_pir_img_from_nifti_img
from SpineSeg.utils import getSpineBBoxFromImage, get_result_of_mask
from SpineSeg.utils import removeAllFiles
import numpy as np
import torch
import time
import nibabel as nib
import SimpleITK as sitk

from utils.file_operations import createSpinePlanningPropFile, createSpineHurFile

torch.set_grad_enabled(False)



def load_models(seg_spine_norm=False, seg_vert_norm=False):

    if seg_spine_norm:
        model_file_seg_binary = 'models/segmentor_spine_norm.pth'
    else:
        model_file_seg_binary = 'models/segmentor_spine.pth'

    if seg_vert_norm:
        model_file_seg_idv = 'models/segmentor_vertebra_norm.pth'
    else:
        model_file_seg_idv = 'models/segmentor_vertebra.pth'

    model_file_loc_sag = 'models/locator_sagittal.pth'
    model_file_loc_cor = 'models/locator_coronal.pth'

    id_group_model_file = 'models/classifier_group.pth'
    id_cer_model_file = 'models/classifier_cervical.pth'
    id_thor_model_file = 'models/classifier_thoracic.pth'
    id_lum_model_file = 'models/classifier_lumbar.pth'


    return {'seg_binary': model_file_seg_binary, 'seg_individual': model_file_seg_idv,
            'loc_sagittal': model_file_loc_sag, 'loc_coronal': model_file_loc_cor,
            'id_group': id_group_model_file, 'id_cervical': id_cer_model_file,
            'id_thoracic': id_thor_model_file, 'id_lumbar': id_lum_model_file}


def spineSeg(input_dcm_dir, output_folder):
    time_sum_start = time.time()
    save_folder = output_folder
    mkpath(save_folder)
    scanname = "CT_0000"
    print(' ... starting to process: ', input_dcm_dir)

    img_data, ori_size, ori_spacing, ori_orient_code, ori_aff, crop_box, dicom_info = getSpineBBoxFromImage(input_dcm_dir,save_folder)
    x_min, x_max, y_min, y_max, z_min, z_max = crop_box
    img_data_trans = img_data.transpose(2, 1, 0)

    nifti_img = nib.Nifti1Image(img_data_trans, ori_aff)

    # =================================================================
    # Load the CT scan
    # =================================================================

    ### TODO: data I/O for other formats

    time_load_start = time.time()
    try:
        pir_img_all = read_isotropic_pir_img_from_nifti_img(nifti_img, ori_spacing, ori_orient_code)
    except ImageFileError:
        sys.exit('The input CT should be in nifti format.')

    y_min_new = int(y_min * ori_spacing[1])
    y_max_new = int(y_max * ori_spacing[1])
    z_min_new = int(z_min * ori_spacing[2])
    z_max_new = int(z_max * ori_spacing[2])
    x_min_new = int(x_min * ori_spacing[0])
    x_max_new = int(x_max * ori_spacing[0])

    pir_img_all_Z, pir_img_all_Y, pir_img_all_X = pir_img_all.shape

    pir_img = pir_img_all[y_min_new:y_max_new, pir_img_all_Y - z_max_new:pir_img_all_Y - z_min_new, pir_img_all_Z-x_max_new:pir_img_all_Z - x_min_new]
    print(' ... loaded CT volume in isotropic resolution and PIR orientation ')
    time_load_end = time.time()

    # =================================================================
    # Spine binary segmentation
    # =================================================================

    models = load_models(seg_spine_norm=False, seg_vert_norm=False)

    time_binary_start = time.time()
    from SpineSeg.segment_spine import binary_segmentor
    binary_mask = binary_segmentor(pir_img, models['seg_binary'], mode='overlap', norm=False)
    time_binary_end = time.time()

    print(' ... obtained spine binary segmentation ')

    # =================================================================
    # Initial locations
    # =================================================================
    time_locate_start = time.time()
    from SpineSeg.locate import locate, locate_yolo
    # locations = locate(pir_img, models['loc_sagittal'], models['loc_coronal'])
    # print("locates ori:", locations)

    pir_img_spine = np.zeros_like(pir_img)
    idx = np.where(binary_mask > 0)
    pir_img_spine[idx] = pir_img[idx]
    locations = locate_yolo(pir_img_spine, binary_mask, output_folder)

    print("locations yolo:", locations)
    #
    #locations = np.array([])
    # loc_file = os.path.join(output_folder, 'loc.npy')
    # locations = np.load(loc_file)
    # print("locations locate:", locations)
    time_locate_end = time.time()
    print(' ... obtained {} initial 3D locations '.format(len(locations)))

    # =================================================================
    # Consistency circle - Locations refine - multi label segmentation
    # =================================================================

    time_con_start = time.time()
    from SpineSeg.consistency_loop import consistency_refinement_close_loop

    multi_label_mask, locations, labels, loc_has_converged = consistency_refinement_close_loop(locations, pir_img,
                                                                                               binary_mask,
                                                                                               models['seg_individual'],
                                                                                               False,
                                                                                               models['id_group'],
                                                                                               models['id_cervical'],
                                                                                               models['id_thoracic'],
                                                                                               models['id_lumbar'])

    print("\nfinall loc:", locations)
    # save_loc_file = os.path.join(output_folder, "loc")
    # np.save(save_loc_file, locations)

    time_con_end = time.time()
    print(' ... obtained PIR multi label segmentation ')

    # =================================================================
    # Save the result in original format
    # =================================================================
    time_save_start = time.time()
    mask = np.zeros_like(pir_img_all)
    mask[y_min_new:y_max_new, pir_img_all_Y - z_max_new:pir_img_all_Y - z_min_new,
    pir_img_all_Z-x_max_new:pir_img_all_Z - x_min_new] = multi_label_mask

    mask = get_result_of_mask(mask, ori_orient_code, ori_spacing, ori_size, ori_aff, save_folder, scanname)

    time_save_end = time.time()
    time_sum_end = time.time()
    print("\n\nload img time:", time_load_end - time_load_start)
    print("locate time:", time_locate_end - time_locate_start)
    print("binary seg time:", time_binary_end - time_binary_start)
    print("con time:", time_con_end - time_con_start)
    print("save time:", time_save_end - time_save_start)
    print("sum time:", time_sum_end - time_sum_start)
    print("\n\n\n")
    return mask, dicom_info



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spine Surgical Planning.')
    parser.add_argument('-i', '--input_dcm_dir', type=str, help='input dicom dir')
    parser.add_argument('-o', '--output_folder', type=str, help='folder to save the results')
    parser.add_argument('-s', '--start_spine_label', type=str,default=None, help='create the spine planning start label')
    parser.add_argument('-e', '--end_spine_label', type=str, default=None, help='create hte spine planning end label')

    args = parser.parse_args()
    # input_dcm_dir = args.input_dcm_dir
    # save_dir = args.output_folder
    start_label = args.start_spine_label
    end_label = args.end_spine_label

    input_dcm_dir = "/media/alg/data3/DeepSpineData/spine_test/Test12/DICOM"
    save_dir = "/media/alg/data3/DeepSpineData/spine_test/Test12/predict_spine"
    mask, dicom_info = spineSeg(input_dcm_dir, save_dir)

    mask_itk = sitk.GetImageFromArray(mask)
    mask_itk.SetOrigin(dicom_info.origin)
    mask_itk.SetSpacing(dicom_info.spacing)
    mask_itk.SetDirection(dicom_info.direction)
    save_mask_file = os.path.join(save_dir, "CT_0000_seg.nii.gz")
    sitk.WriteImage(mask_itk, save_mask_file)

    save_hur_dir = os.path.join(save_dir, "hur")
    os.makedirs(save_hur_dir, exist_ok=True)
    removeAllFiles(save_hur_dir)

    createSpinePlanningPropFile(mask, dicom_info, save_hur_dir)

    patient_id = dicom_info.patientID
    createSpineHurFile(save_hur_dir, save_dir, patient_id)


