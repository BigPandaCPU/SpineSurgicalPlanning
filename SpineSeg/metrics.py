# -*- coding: utf-8 -*-
__copyright__ = \
"""
Copyright &copyright © (c) 2021 Inria Grenoble Rhône-Alpes.
All rights reserved.

This source code is to be used for academic research purposes only.
For commercial uses of the code, please send an email to edmond.boyer@inria.fr and sergi.pujades@inria.fr

"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Di Meng, Anjany Sekuboyina"


import numpy as np
import SimpleITK as sitk

def compute_dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    # print('dice: ', 2. * intersection.sum() / im_sum)
    return 2. * intersection.sum() / im_sum


def compute_multi_dice(prediction, gt):

    '''
    Evaluate the segmentatio with multi labels.
    For each class, compute a single DSC, and take the mean over all the classes.
    '''

    labels = sorted(np.unique(gt))
    num_labels = len(labels)

    assert num_labels >= 1

    print('{} labels in the gt mask: {}'.format(num_labels-1, labels[1:]))

    dsc = 0

    if num_labels == 1:
        print('No label in the ground truth.')
        return dsc 
    else:
        for l in range(1, num_labels):

            gt_l = np.copy(gt)
            gt_l[gt_l!=labels[l]] = 0
            gt_l[gt_l > 0] = 1

            pred_l = np.copy(prediction)
            pred_l[pred_l!=labels[l]] = 0
            pred_l[pred_l > 0] = 1

            dsc += compute_dice(pred_l, gt_l)
            print('DSC label {}: {}'.format(labels[l], DSC(pred_l, gt_l)))
            # print(dsc)
        dsc = dsc/(num_labels-1)
        print('Multi label DSC: ', dsc)
        return dsc


def compute_hd(im_gt_array, im_pred_array, spacing=[1,1,1]):
    """
    Computes Hausdorff, a measure of set similarity.
    Parameters
    ----------
    im_gt : itk_image with boolean values.
        
    im_pred : itk_image with boolean values.

    Returns
    -------
    hd : float
        If prediction is empty (while ground truth isn't) = np.inf
        
    Notes
    -----
    Make sure the spacing of im_gt and im_pred are same and correct. 
    SimpleITK comsiders this spacing to compute HD. 
    """

    if im_pred_array.sum() == 0:
        hd = 100
    else:

        im_gt = sitk.GetImageFromArray(im_gt_array)
        im_pred = sitk.GetImageFromArray(im_pred_array)

        im_gt.SetSpacing(spacing)
        im_pred.SetSpacing(spacing)

        hd_filter = sitk.HausdorffDistanceImageFilter()            
        hd_filter.Execute(im_gt, im_pred)
        hd = hd_filter.GetHausdorffDistance()
    
    return hd


def compute_multi_hd(prediction, gt, spacing=[1,1,1]):

    labels = sorted(np.unique(gt))
    num_labels = len(labels)

    assert num_labels > 1

    print('{} labels in the gt mask: {}'.format(num_labels-1, labels[1:]))

    hd = 0

    for l in range(1, num_labels):

        gt_l = np.copy(gt)
        gt_l[gt_l!=labels[l]] = 0
        gt_l[gt_l > 0] = 1
        gt_l = get_largest_component(gt_l)
        # gt_l = gt_l.astype(np.bool)

        pred_l = np.copy(prediction)
        pred_l[pred_l!=labels[l]] = 0
        pred_l[pred_l > 0] = 1
        pred_l = get_largest_component(pred_l)
        # pred_l = pred_l.astype(np.bool)

        hd_l = compute_hd(gt_l, pred_l, spacing)
        hd += hd_l
        print('HD label {}: {}'.format(labels[l], hd_l))
        # print(dsc)
    hd = hd/(num_labels-1)
    print('Multi label HD: ', hd)
    return hd



def get_largest_component(pred):

    '''find the largest connected component and return in binary'''
    import numpy as np 
    from scipy.ndimage.measurements import label

    structure = np.ones((3, 3, 3), dtype=np.int)

    labeled, ncomponents = label(pred, structure)

    if ncomponents > 1:
        labels = np.unique(labeled)[1:]

        label_to_keep = 1
        largest_count = 0

        for label in labels:
            num_label = np.count_nonzero(labeled==label)
            # print('component {}: {}'.format(label, num_label))

            if num_label > largest_count:
                label_to_keep = label 

        labeled[labeled!=label_to_keep] = 0

    labeled[labeled>0] = 1

    assert labeled.shape == pred.shape

    return labeled


def reorganize_annos(annos):

    num_labels = len(annos)

    new_annos = dict()

    for anno in annos:

        label = anno['label']
        x = anno['X']
        y = anno['Y']
        z = anno['Z']

        new_annos[str(label)] = [x, y, z]

    return new_annos


def dist(x1, y1, z1, x2, y2, z2):
    import numpy as np 

    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)


def verse_metrics(gt_seg_file, gt_ctd_file,
                  pred_seg_file, pred_ctd_file):

    ## return Dice, ID rate, MSE, HD

    
    from utils import read_json_file, read_nifti_file, get_size_and_spacing_and_orientation_from_nifti_file
    import pandas as pd
    import numpy as np 

    gt_mask = read_nifti_file(gt_seg_file)
    gt_annos = read_json_file(gt_ctd_file)

    pred_mask = read_nifti_file(pred_seg_file)
    pred_annos = read_json_file(pred_ctd_file)
    pred_annos = reorganize_annos(pred_annos)

    _, spacing, _, _ =  get_size_and_spacing_and_orientation_from_nifti_file(gt_seg_file)
    spacing = spacing.astype(np.double)


    metric = dict()
    num_positive_id = 0
    num_negative_id = 0
    mse_list = []
    hd_list = []
    dsc_list = []

    # ----------ANNOTATION METRICS---------------------

    for idx, gt_anno in enumerate(gt_annos):
        
        gt_label = gt_anno['label']
        gt_x = gt_anno['X']
        gt_y = gt_anno['Y']
        gt_z = gt_anno['Z']

        metric[str(gt_label)] = []

        if str(gt_label) in pred_annos.keys():

            pred_x, pred_y, pred_z = pred_annos[str(gt_label)]
            mse = dist(pred_x, pred_y, pred_z, gt_x, gt_y, gt_z)

            if mse <= 20:
                metric[str(gt_label)].append(1)
                num_positive_id += 1
            else:
                metric[str(gt_label)].append(0)
                num_negative_id += 1

            metric[str(gt_label)].append(mse)

            mse_list.append(mse)


        else:

            metric[str(gt_label)].append(0)
            metric[str(gt_label)].append(None)

            num_negative_id += 1
            mse_list.append(None)


    # ----------SEGMENTATION METRICS --------------------

        pred_mask_label = (pred_mask == gt_label)
        gt_mask_label = (gt_mask == gt_label)

        dsc = compute_dice(pred_mask_label, gt_mask_label)

        if gt_label not in pred_mask:
            hd = None
        else:
            hd = compute_hd(pred_mask_label.astype(np.int), gt_mask_label.astype(np.int), spacing)

        metric[str(gt_label)].append(dsc)
        metric[str(gt_label)].append(hd)

        dsc_list.append(dsc)
        hd_list.append(hd)


    mse_list = [mse for mse in mse_list if mse is not None]
    hd_list = [hd for hd in hd_list if hd is not None]
    dsc_list = [dsc for dsc in dsc_list if dsc is not None]

    metric['mean'] = []

    id_rate = num_positive_id / (num_positive_id + num_negative_id)
    metric['mean'].append(id_rate)

    mean_mse = np.nanmean(np.array(mse_list))
    std_mse = np.nanstd(np.array(mse_list))

    metric['mean'].append(mean_mse)

    mean_dsc = np.nanmean(np.array(dsc_list))
    std_dsc = np.nanstd(np.array(dsc_list))

    metric['mean'].append(mean_dsc)

    mean_hd = np.nanmean(np.array(hd_list))
    std_hd = np.nanstd(np.array(hd_list))

    metric['mean'].append(mean_hd)

    metric_out = pd.DataFrame.from_dict(metric, orient='index', 
                                                columns=['Id rate', 'MLD', 'Dice', 'HD'])

    return metric_out
