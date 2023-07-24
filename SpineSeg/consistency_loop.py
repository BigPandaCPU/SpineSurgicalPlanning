__copyright__ = \
"""
Copyright &copyright © (c) 2021 Inria Grenoble Rhône-Alpes.
All rights reserved.

This source code is to be used for academic research purposes only.
For commercial uses of the code, please send an email to edmond.boyer@inria.fr and sergi.pujades@inria.fr

"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Di Meng"



def check_locs_updated(prev_locations, locations, loc_threshold=1.5):
    import numpy as np 

    if len(locations) == 0:
        return False

    if len(prev_locations) != len(locations):
        return False

    dists = np.linalg.norm(prev_locations - locations, axis=1)
    if np.max(dists) > loc_threshold:
        return False

    return True

def check_labels_updated(prev_labels, labels, label_threshold=1):
    import numpy as np 

    if len(prev_labels) != len(labels):
        return False
    
    if None in labels or None in prev_labels:
        return False
    
    dists = np.linalg.norm(prev_labels - labels, axis=0)
    if np.max(dists) > label_threshold:
        return False

    return True

def check_masks_updated(prev_ind_mask, idv_mask_list, mask_threshold=0.95):
    import numpy as np 


    if len(prev_ind_mask) != len(idv_mask_list):
        return False

    for mask1, mask2 in zip(prev_ind_mask, idv_mask_list):
        pct = masks_overlapping_pct(mask1, mask2)

        if pct < mask_threshold:
            return False

    return True


def compute_dist(loc1, loc2):
    import numpy as np 
    x1, y1, z1 = loc1[:]
    x2, y2, z2 = loc2[:]

    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)


def masks_overlapping_pct(mask1, mask2):
    import numpy as np 

    if mask1 is None:
        return 0

    intersection = np.logical_and(mask1==1, mask2==1).astype(np.int)
    union = np.logical_or(mask1==1, mask2==1).astype(np.int)
    intersection_count = np.count_nonzero(intersection)
    union_count = np.count_nonzero(union)
    if union_count == 0:
        return 0
    else:
        return intersection_count/union_count


def loc_and_msk_convergence(loc_iter_list, loc_dist_list, msk_pct_list):

    if not loc_dist_list or not msk_pct_list:
        return False

    else:
        loc_dist = loc_dist_list[-1]
        msk_pct = msk_pct_list[-1]


        if abs(loc_dist) == 0 or msk_pct == 1 : 
            print('Convergence obtained. ')
            return True

            
        current_loc = loc_iter_list[-1]

        for i in range(2, len(loc_iter_list)+1):
            if (loc_iter_list[-i] == current_loc).all():
                print('Oscillation detected in {} round.'.format(i-1))
                return True

        print('Convergence failed.')
        return False


def duplicated_locations_index(individual_mask_list): 
    import numpy as np 

    duplicated_locations_idx = []

    for i in range(len(individual_mask_list)-1):
        current_mask = individual_mask_list[i]
        next_mask = individual_mask_list[i+1]

        overlapping = np.logical_and(current_mask==1, next_mask==1).astype(np.int)

        count_overlapping = np.count_nonzero(overlapping)
        current_mask_count = np.count_nonzero(current_mask)

        if current_mask_count == 0:
            duplicated_locations_idx.append(i)
        else:
            overlap_pct = masks_overlapping_pct(current_mask, next_mask)


            if overlap_pct > 0.5:
                duplicated_locations_idx.append(i)
            print('count_overlapping: {}, percentage: {}'.format(count_overlapping, overlap_pct))

    print('duplicated_locations_idx: ', duplicated_locations_idx)

    return duplicated_locations_idx


def remove_duplicated_locations_with_masks(location_list, mask_list, 
                                           loc_has_converged_list, loc_needs_refinement_list, 
                                           idx_list):
    # remove duplicated locations and masks by index and sort in order
    import numpy as np 

    if not idx_list:

        locations = np.array(location_list)

        sort_idx = np.argsort(locations[:,1])
        locations = locations[sort_idx]

        loc_has_converged= np.array(loc_has_converged_list)
        loc_has_converged = loc_has_converged[sort_idx]

        loc_needs_refinement= np.array(loc_needs_refinement_list)
        loc_needs_refinement = loc_needs_refinement[sort_idx]

        masks = []
        for i in sort_idx:
            masks.append(mask_list[i])
        return locations, masks, loc_has_converged.tolist(), loc_needs_refinement.tolist()

    else:
        for idx in sorted(idx_list, reverse=True):
            del location_list[idx]
            del mask_list[idx]
            del loc_has_converged_list[idx]

        locations = np.array(location_list)
        sort_idx = np.argsort(locations[:,1])
        locations = locations[sort_idx]

        loc_has_converged= np.array(loc_has_converged_list)
        loc_has_converged = loc_has_converged[sort_idx]

        loc_needs_refinement= np.array(loc_needs_refinement_list)
        loc_needs_refinement = loc_needs_refinement[sort_idx]

        masks = []
        for i in sort_idx:
            masks.append(mask_list[i])
        return locations, masks, loc_has_converged.tolist(), loc_needs_refinement.tolist()



def locations_and_masks_without_duplication(loc_list, mask_list, loc_has_converged, loc_needs_refinement):
    # remove duplicated locations based on the masks overlap
    import numpy as np 

    duplicated_locations_idx = duplicated_locations_index(mask_list)

    locations, mask_list, loc_has_converged, loc_needs_refinement = remove_duplicated_locations_with_masks(loc_list, mask_list, 
                                                                                        loc_has_converged, loc_needs_refinement,
                                                                                        duplicated_locations_idx)
    locations = np.array(locations).astype(np.int)

    return locations, mask_list, loc_has_converged, loc_needs_refinement


def individual_binary_mask_from_list(mask_list):
    # assembling the individual masks into one binary mask 
    import numpy as np 
    individual_binary_mask = np.copy(mask_list[0])

    for mask in mask_list[1:]:
        if mask is not None:
            individual_binary_mask += mask 

    individual_binary_mask[individual_binary_mask > 1] = 1

    return individual_binary_mask


def per_location_refiner_iter(loc, pir_img, model_file_seg_idv, seg_idv_norm,
                                binary_mask, max_iter):

    from SpineSeg.segment_vertebra import per_location_refiner_segmentor
    import numpy as np 

    converged = False

    loc_iter_list = [loc, ]
    mask_iter_list = [None, ]

    loc2pre_list = []
    msk2pre_list = []

    loop_iter = 0
    while True:

        loop_iter += 1

        x, y, z = loc[:]
        loc, mask = per_location_refiner_segmentor(x, y, z, pir_img, model_file_seg_idv, seg_idv_norm)

        if loc is None:
            loc_iter_list.append(None)
            mask_iter_list.append(None)
            break


        loc_iter_list.append(loc)
        mask_iter_list.append(mask)

        assert len(loc_iter_list) > 1

        loc_pre = loc_iter_list[-2]
        dist_to_pre = compute_dist(loc_pre, loc)
        loc2pre_list.append(dist_to_pre)

        assert len(mask_iter_list) > 1
            
        msk_pre = mask_iter_list[-2]
        msk_to_pre_pct = masks_overlapping_pct(msk_pre, mask)
        msk2pre_list.append(msk_to_pre_pct)

        ## mask sure the location is in spine view - remove the false detections from initial locations
        in_bi = True
        if binary_mask is not None:
            overlap_w_bi = np.logical_and(binary_mask, mask).astype(np.int)
            overlap_w_bi_per = np.count_nonzero(overlap_w_bi)/np.count_nonzero(mask)
            print('segmented mask overlapped {:02f}% with binary mask'.format(100.*overlap_w_bi_per))
            in_bi = False if overlap_w_bi_per < 0.4 else True


        if loc_and_msk_convergence(loc_iter_list, loc2pre_list, msk2pre_list) and in_bi:
            converged = True
            break

        if max_iter is not None and loop_iter >= max_iter:
            loc_iter_list[-1] = None
            mask_iter_list[-1] = None
            print('Discard the unstable location. ')
            break


    return loc_iter_list[-1], mask_iter_list[-1], converged


def locations_refiner_iter(loc_list, mask_list, labels, pir_img, binary_mask,
                           loc_needs_refinement, loc_has_converged,
                           model_file_seg_idv, seg_idv_norm, max_iter=None):

    import numpy as np 

    assert len(loc_list) == len(labels)
    assert len(loc_list) == len(mask_list)

    new_locations = []
    new_idv_mask = []

    for i, loc in enumerate(loc_list):
        print('\n ========== refining location {}/{}, label: {} ============'.format(i+1, len(loc_list), labels[i]))

        if not loc_needs_refinement[i]:
            new_locations.append(loc)
            new_idv_mask.append(mask_list[i])            
            continue 


        loc, mask, converged = per_location_refiner_iter(loc, pir_img, model_file_seg_idv, seg_idv_norm,
                                                            binary_mask, max_iter=max_iter)

        if loc is None:
            loc_needs_refinement[i] = None
            loc_has_converged[i] = None
            continue 

        new_locations.append(loc)
        new_idv_mask.append(mask)

        loc_has_converged[i] = converged
        loc_needs_refinement[i] = True if labels[i] is None else False

    while None in loc_has_converged:
        loc_needs_refinement.remove(None)
        loc_has_converged.remove(None)

    assert len(new_locations) == len(new_idv_mask) == len(loc_needs_refinement) == len(loc_has_converged)

    return new_locations, new_idv_mask


def labels_to_group_labels(labels):
    import numpy as np 

    if isinstance(labels, int):
        if labels <=7:
            labels = 0
        elif labels > 7 and labels < 20:
            labels = 1
        else:
            labels = 2
        return labels 

    else:
        group_labels = []
        for label in labels:
            if label <=7:
                label = 0
            elif label > 7 and label < 20:
                label = 1
            else:
                label = 2
            group_labels.append(label)

        return np.array(group_labels)


def load_vertebrae_sizes_pkl(pkl_file='SpineSeg/statistics/vertebrae_sizes.pkl'):
    import pickle 

    file = open(pkl_file, 'rb')
    data = pickle.load(file)

    return data 


def _in_frange(x, low_bound, high_bound):

    if x >= low_bound and x <= high_bound:
        return 1

    elif x < low_bound:
        import warnings
        warnings.warn('Narrow disc detected. ')
        return 0

    else:
        return -1


def filtered_connected_components(diff, 
                                  locations, labels, mask_list,
                                  threshold_ratio=0.5,
                                  threshold=7820):
    import numpy as np 
    from scipy.ndimage.measurements import label as scipy_label
    from scipy.ndimage.measurements import center_of_mass

    structure = np.ones((3, 3, 3), dtype=np.int)

    components_labels, ncomponents = scipy_label(diff, structure)

    components = []
    cc_labels = []


    if len(locations) == 0:
        for l in range(1, ncomponents+1):
            component = (components_labels==l)
            count = np.count_nonzero(component)

            if count > threshold * threshold_ratio:
                components.append(component)
                cc_labels.append(None)

        return components, cc_labels


    for l in range(1, ncomponents+1):

        component = (components_labels==l)
        count = np.count_nonzero(component)
        # print('component count: ', count)
        if count <= threshold * threshold_ratio:
            continue

        ctd = center_of_mass(component)
        dist_to_locations = np.linalg.norm(locations - ctd, axis=1)
        assert len(dist_to_locations) == len(locations)

        closest = np.argmin(dist_to_locations)


        label = labels[closest]
        mask = mask_list[closest]
        size_closest = np.count_nonzero(mask==1)
        group_label = labels_to_group_labels(label)

        
        # To get the size of the closest location vertebra
        #    check if the size of component is in the distribution (mean , std)
        #    if not : use min size 
        #    if yes: , use the current size, 
        vertebrae_size_dict = load_vertebrae_sizes_pkl()
        label = 19 if label == 28 else label 
        size_mean = vertebrae_size_dict[str(label)]['mean']
        size_std = vertebrae_size_dict[str(label)]['std']
        size_min = vertebrae_size_dict[str(label)]['min']
        size_max = vertebrae_size_dict[str(label)]['max']


        print('closest vertebra size: ', size_closest)
        print('closest vertebra size mean: {}, std: {}, min: {}, max: {}'.format(
                                        size_mean, size_std, size_min, size_max))

        if _in_frange(size_closest, size_mean-3*size_std, size_mean+3*size_std) != 1:
            size_closest = size_min 
        else:
            size_closest = size_closest

        print('used closest vertebra size: ', size_closest)

        # compute sign distances to the locations : y axis
        # this should be coherent with the closest index -> 0 backwards, len-1 forward
        # use the size and sign for choosing the ratio to pre or ratio to next
        # predicted_vertebra_size = 7820
        closest_loc = locations[closest]
        _, closest_loc_y, _ = closest_loc[:]
        _, component_y, _ = ctd[:]

        signed_direc = component_y - closest_loc_y
        if signed_direc > 0:
            direc = 'next'
            component_label = label + 1 if label < 25 else label
        elif signed_direc < 0:
            direc = 'pre'
            component_label = label - 1 if label >1 else label
        else:
            import warnings 
            warnings.warn('overlapped y axis of componet and the closest vertebra.')
            direc = 'pre'
            component_label = label

        print('component detected in [{}] of label {}.'.format(direc, label))
        print('component label: ', component_label)

        from SpineSeg.regressor import regressor_size_one_side
        predicted_vertebra_size = regressor_size_one_side(size_closest, direc, level=group_label)
        
        print('component count: ', count)

        # predicted_vertebra_size = threshold
        print('component threshold: ', predicted_vertebra_size)


        if count > predicted_vertebra_size * threshold_ratio:
            print('component is the missing vertebra.')
        
            components.append(component)
            cc_labels.append(component_label)

    cc_labels = np.array(cc_labels)

    return components, cc_labels


def get_extra_locations_from_components(components, cc_labels, locations):
    import numpy as np 
    from scipy.ndimage.measurements import center_of_mass

    extra_centers = []
    extra_labels = []

    loc_list = locations.tolist()
    unique_locations = []
    for loc in loc_list:
        if loc not in unique_locations:
            unique_locations.append(loc)

    locations = np.array(loc_list)

    ## add two locations per time to accelerate the process when there's few detections
    # if len(locations) > 0 and len(locations) <= 3:
    if len(locations) == 1 and len(components) == 1:
        for component, cc_label in zip(components, cc_labels):
            x_array, y_array, _z_array = np.where(component>0)

            ctd = center_of_mass(component)

            print('ctd: ', ctd)
            extra_centers.append(ctd)
            extra_labels.append(cc_label)

            count = np.count_nonzero(component)
            print('== size of the component: ', count)

            if count < 30000:
                continue

            h,w,d = component.shape

            for loc in unique_locations:
                x,y,z = loc[:]
                x,y,z = int(x), int(y), int(z)

                if y > min(y_array) and y < max(y_array):

                    print('component longitude: {} - {}'.format(min(y_array), max(y_array)))
                    print('existing location: ', loc)

                    print('cutting the component into two.')
                    component_1 = np.zeros(component.shape)
                    component_2 = np.zeros(component.shape)
                    component_1[:,0:y,:] = component[:,0:y,:]
                    component_2[:,y:w,:] = component[:,y:w,:]

                    ctd_1 = center_of_mass(component_1)
                    print('ctd_1: ', ctd_1)
                    ctd_2 = center_of_mass(component_2)
                    print('ctd_2: ', ctd_2)

                    extra_centers.append(ctd_1)
                    extra_labels.append(cc_label)
                    extra_centers.append(ctd_2)
                    extra_labels.append(cc_label)

    ## add locations one by one - regular approach 
    else:
        for component, cc_label in zip(components, cc_labels):
            ctd = center_of_mass(component)
            extra_centers.append(ctd)
            extra_labels.append(cc_label)

    # extra_centers = np.array(extra_centers)

    return extra_centers, extra_labels


def extra_locations_refinement(extra_locations, locations, extra_labels, pir_img, binary_mask,
                                idv_mask_list, model_file_seg_idv, seg_idv_norm, fishing=False):

    import numpy as np 

    extra_masks = []
    extra_locs = []
    extra_refinement = []
    extra_convergence = []

    for i, (loc, extra_label) in enumerate(zip(extra_locations, extra_labels)):
        converged = [False]
        x, y, z = loc[:]
        new_loc_list, mask_list = locations_refiner_iter([loc], [None], [extra_label], pir_img, binary_mask,
                                                    [True], converged, model_file_seg_idv, seg_idv_norm, max_iter=10)

        if len(new_loc_list) == 0:
            if not fishing:
                continue 
            else:
                if  extra_label !=25:  # if we go for fishing, we keep the vertebrae till L5
                    
                    psudo_mask = np.zeros(pir_img.shape)
                    psudo_mask[x,y,z] = 1
                    extra_masks.append(psudo_mask)
                    extra_locs.append(loc)
                    extra_refinement.append(False)
                    extra_convergence.append(False)

                    print('unstable location label {} at {}, adding it'.format(loc, extra_label))

                    return extra_masks, extra_locs, extra_refinement, extra_convergence
                else:  # L6 not found, we don't add it

                    print('unstable location L6, discarding it')
                    continue 


        for new_loc, mask in zip(new_loc_list, mask_list):

            if new_loc is None:
                continue 

            # if the extra locs are not duplicated, then add 
            if extra_mask_valid(new_loc, mask, locations, idv_mask_list):
                extra_masks.append(mask)
                extra_locs.append(new_loc)
                extra_refinement.append(False)
                extra_convergence.append(converged[0])

            else:
                from SpineSeg.segment_vertebra import per_location_refiner_segmentor
                new_loc, mask = per_location_refiner_segmentor(x, y, z, pir_img, model_file_seg_idv, seg_idv_norm)

                if extra_mask_valid(new_loc, mask, locations, idv_mask_list) and extra_label == 1:
                    print('adding one shot mask for {}.'.format(extra_label))
                    extra_masks.append(mask)
                    extra_locs.append(new_loc)
                    extra_refinement.append(False)
                    extra_convergence.append(False)
                else:
                    print('oneshot mask discarded.')


    return extra_masks, extra_locs, extra_refinement, extra_convergence


def update_locations_masks_labels(extra_masks, extra_locs, extra_refinement, extra_convergence,
                                   locations, idv_mask_list, loc_needs_refinement, loc_has_converged,
                                   sliding_mask, binary_mask, model_file_id_group, model_file_id_cer,
                                    model_file_id_thor, model_file_id_lum):
    import numpy as np 
    from SpineSeg.identify import labelling_2msk
    

    if len(locations) == 0:
        idv_mask_list = extra_masks
        loc_needs_refinement = extra_refinement
        loc_has_converged = extra_convergence
        locations = np.array(extra_locs)
    else:
        idv_mask_list = idv_mask_list + extra_masks
        loc_needs_refinement = loc_needs_refinement + extra_refinement
        loc_has_converged = loc_has_converged + extra_convergence
        locations = np.concatenate((locations, np.array(extra_locs)), 0).astype(np.int)

    resorting_indices = locations[:,1].argsort()
    locations = locations[resorting_indices]
    idv_mask_list = [idv_mask_list[i] for i in resorting_indices]
    loc_needs_refinement = [loc_needs_refinement[i] for i in resorting_indices]
    loc_has_converged = [loc_has_converged[i] for i in resorting_indices]


    individual_binary_mask = individual_binary_mask_from_list(idv_mask_list)

    binary_mask = np.logical_or(binary_mask, individual_binary_mask).astype(np.int)

    labels = labelling_2msk(sliding_mask, individual_binary_mask, locations, loc_has_converged, 
                            model_file_id_group, model_file_id_cer, model_file_id_thor, model_file_id_lum)

    assert len(locations) == len(labels) == len(idv_mask_list)

    # print('add {} extra locations. '.format(len(extra_masks)))

    return locations, labels, idv_mask_list, individual_binary_mask, binary_mask, loc_needs_refinement, loc_has_converged
    

def get_group_interval_distribution(coeff_pkl_file='SpineSeg/statistics/vertebrae_interval_distributions_coeff.pkl'):
    import pickle

    file = open(coeff_pkl_file, 'rb')
    coeff = pickle.load(file)

    cervical_coeff = coeff['cervical']
    thoracic_coeff = coeff['thoracic']
    lumbar_coeff = coeff['lumbar']

    return cervical_coeff, thoracic_coeff, lumbar_coeff


def _is_gap_fitting_gauss_distribution(dist, label, cervical_coeff, thoracic_coeff, lumbar_coeff):

    assert label in [0, 1, 2]

    mean = 0
    sigma = 0

    if label == 0:
        mean, sigma = cervical_coeff[:]
    elif label == 1:
        mean, sigma = thoracic_coeff[:]
    elif label == 2:
        mean, sigma = lumbar_coeff[:]
    else:
        raise ValueError('Wrong label value {}'.format(label))

    assert mean != 0 and sigma != 0

    distribute_coeff = (dist - mean) / sigma 

    if distribute_coeff > 4 and _in_frange(dist, mean-3*sigma, mean+3*sigma) < 0:
        print('distribution NOT fit, label: {}, dist: {}, mean: {}, sigma: {}'.format(label, dist, mean, sigma))

        return [0, mean]

    print('distribution fit, label: {}, dist: {}, mean: {}, sigma: {}'.format(label, dist, mean, sigma))
    return [1, -1]


def _is_gap_fitting_regressor(gap_list, idx, group_label):
    from SpineSeg.regressor import regressor_gap_one_side, regressor_gap_two_sides
    from sklearn.metrics import mean_absolute_error
    import numpy as np 

    level_label = group_label + 1

    gt_gap = gap_list[idx]

    if idx == 0:
        if level_label == 1:
            mre_mean = 12.13
            mre_std = 3.36

            mae_mean = 1.90
            mae_std = 0.62

        elif level_label == 2:
            mre_mean = 4.17
            mre_std = 0.88

            mae_mean = 1.05
            mae_std = 0.17
            
        elif level_label == 3:
            mre_mean = 5.16
            mre_std = 1.71

            mae_mean = 1.68
            mae_std = 0.56
            
        else:
            raise NotImplementedError('Wrong group label {}, it should be 1 or 2 or 3.'.format(level_label))
        pred_gap = regressor_gap_one_side(gap_list[idx+1], 'next', level=level_label)

    elif idx == (len(gap_list)-1):
        if level_label == 1:
            mre_mean = 10.2
            mre_std = 2.05

            mae_mean = 1.43
            mae_std = 0.22
            
        elif level_label == 2:
            mre_mean = 3.96
            mre_std = 1.56

            mae_mean = 1.05
            mae_std = 0.43
            
        elif level_label == 3:
            mre_mean = 4.45
            mre_std = 1.55

            mae_mean = 1.39
            mae_std = 0.48
            
        else:
            raise NotImplementedError('Wrong group label {}, it should be 1 or 2 or 3.'.format(level_label))
        pred_gap = regressor_gap_one_side(gap_list[idx-1], 'pre', level=level_label)

    else:
        if level_label == 1:
            mre_mean = 9.13
            mre_std = 2.86

            mae_mean = 1.31
            mae_std = 0.35
            
        elif level_label == 2:
            mre_mean = 2.42
            mre_std = 1.43

            mae_mean = 0.64
            mae_std = 0.38
            
        elif level_label == 3:
            mre_mean = 2.04
            mre_std = 0.93

            mae_mean = 0.65
            mae_std = 0.27
            
        else:
            raise NotImplementedError('Wrong group label {}, it should be 1 or 2 or 3.'.format(level_label))
        pred_gap = regressor_gap_two_sides(gap_list[idx-1], gap_list[idx+1], level=level_label)


    mae_pred = np.abs(gt_gap - pred_gap)
    mre_pred = np.mean(np.abs((gt_gap - pred_gap) / gt_gap)) * 100

    fit_mae = True
    fit_mre = True

    if (mae_pred < mae_mean - 3*mae_std) or (mae_pred > mae_mean + 3*mae_std):
        fit_mae = False

    if (mre_pred < mre_mean - 3*mre_std) or (mre_pred > mre_mean + 3*mre_std):
        fit_mre = False

    if not fit_mae and not fit_mre:
        print('return false')
        return [0, pred_gap]

    print('return true')
    return [1, -1]


def extra_locations_in_gaps(loc1, loc2, search_dist, num_extra_loc):
    import numpy as np 

    d12 = compute_dist(loc1, loc2)

    x1, y1, z1 = loc1[:]
    x2, y2, z2 = loc2[:]

    delta_x = int(search_dist*(x2-x1)/d12)
    delta_y = int(search_dist*(y2-y1)/d12)
    delta_z = int(search_dist*(z2-z1)/d12)

    extra_locs = []

    for i in range(num_extra_loc):
        xs, ys, zs = x1+(i+1)*delta_x, y1+(i+1)*delta_y, z1+(i+1)*delta_z 
        extra_locs.append([xs, ys, zs])

    return extra_locs


def get_extra_locations_from_intervals(locations, labels):
    import numpy as np 
    from SpineSeg.regressor import regressor_gap_one_side, regressor_gap_two_sides

    assert len(locations) >= 3

    cervical_coeff, thoracic_coeff, lumbar_coeff = get_group_interval_distribution()

    group_labels = labels_to_group_labels(labels)

    extra_locs = []
    extra_labels = []

    gap_list = []


    loc_list = locations.tolist()
    unique_locations = []
    for loc in loc_list:
        if loc not in unique_locations:
            unique_locations.append(loc)


    for i, loc in enumerate(unique_locations[:-1]):
        current_loc = loc 
        next_loc = unique_locations[i+1]

        dist = compute_dist(current_loc, next_loc)

        gap_list.append(dist)

    for i, gap in enumerate(gap_list):

        label = labels[i]
        group_label = group_labels[i]  

        if_fit_distr = _is_gap_fitting_gauss_distribution(gap, group_label, cervical_coeff, thoracic_coeff, lumbar_coeff)

        # regressor to predict the current gap using pre and next
        if_fit_regressor = _is_gap_fitting_regressor(gap_list, i, group_label)
        print('if fit regressor: ', if_fit_regressor)


        if not if_fit_distr[0] and if_fit_regressor[0]:

            pred_gap = if_fit_distr[1]
            num_extra_loc = int(gap//pred_gap)
            search_dist = int(gap/(num_extra_loc+1))

            current_loc = unique_locations[i]
            next_loc = unique_locations[i+1]

            extra_locs_gap = extra_locations_in_gaps(current_loc, next_loc, search_dist, num_extra_loc)
            extra_labels_gap = [label] * len(extra_locs_gap)

            extra_locs = extra_locs + extra_locs_gap
            extra_labels = extra_labels + extra_labels_gap

            assert len(extra_locs) == len(extra_labels)

        elif not if_fit_distr[0] and not if_fit_regressor[0]:

            pred_gap = if_fit_regressor[1]
            num_extra_loc = int(gap//pred_gap)
            search_dist = int(gap/(num_extra_loc+1))

            current_loc = unique_locations[i]
            next_loc = unique_locations[i+1]

            extra_locs_gap = extra_locations_in_gaps(current_loc, next_loc, search_dist, num_extra_loc)
            extra_labels_gap = [label] * len(extra_locs_gap)

            extra_locs = extra_locs + extra_locs_gap
            extra_labels = extra_labels + extra_labels_gap

            assert len(extra_locs) == len(extra_labels)

        else:
            pass

    return extra_locs, extra_labels


def fish_up(loc1, loc2):
    import numpy as np 

    d12 = compute_dist(loc1, loc2)

    x1, y1, z1 = loc1[:]
    x2, y2, z2 = loc2[:]

    delta_x = int(x2-x1)
    delta_y = int(y2-y1)
    delta_z = int(z2-z1)

    xf, yf, zf = x1-delta_x, y1-delta_y, z1-delta_z

    if min(xf, yf, zf) > 0:
        return np.array([xf, yf, zf])
    else:
        return None


def fish_down(loc1, loc2, vol_size):
    import numpy as np 

    d12 = compute_dist(loc1, loc2)

    x1, y1, z1 = loc1[:]
    x2, y2, z2 = loc2[:]

    delta_x = int(x2-x1)
    delta_y = int(y2-y1)
    delta_z = int(z2-z1)

    xl, yl, zl = x2+delta_x, y2+delta_y, z2+delta_z 

    if xl < vol_size[0] and yl < vol_size[1] and zl < vol_size[2]:
        return np.array([xl, yl, zl])
    else:
        return None 



def extra_mask_valid(new_loc, extra_mask, locations, mask_list):
    
    if len(locations) == 0:
        return True

    for loc, mask in zip(locations, mask_list):
        dist = compute_dist(loc, new_loc)

        # check the dist to existing loc, futher than 40mm, valid 
        if dist < 40:

            iou = masks_overlapping_pct(extra_mask, mask)
            print('check extra mask valid, iou: ', iou)

            if iou > 0.5:
                return False

    return True



def clean_predictions_in_sacrum(idv_msk_list, labels):
    import numpy as np

    if isinstance(labels, list):
        labels = np.array(labels)

    # when there is T13, check the second largest label
    if labels.max() == 28:
        max_label_idx = np.where(labels == np.sort(labels)[-2])[0][0]
    else:
        max_label_idx = np.where(labels == np.sort(labels)[-1])[0][0]

    if max_label_idx != (len(labels)-1):
        idv_msk_list = [idv_msk_list[l] for l in range(max_label_idx+1)]
        labels = [labels[l] for l in range(max_label_idx+1)]

    return idv_msk_list, labels 



def aggregate_multi_label_segmentation(idv_msk_list, labels):
    import numpy as np 

    
    multi_label_mask = np.zeros(idv_msk_list[0].shape)

    assert len(idv_msk_list) == len(labels)

    idv_msk_list, labels = clean_predictions_in_sacrum(idv_msk_list, labels)

    for label, mask in zip(labels, idv_msk_list):
        multi_label_mask[mask == 1] = label


    print('labels in the multi_label_mask: ', np.unique(multi_label_mask))

    return multi_label_mask


def consistency_refinement_close_loop(locations, pir_img, binary_mask,
                                      model_file_seg_idv, seg_idv_norm,
                                      model_file_id_group, model_file_id_cer,
                                      model_file_id_thor, model_file_id_lum):

    import numpy as np
    import time
    from SpineSeg.identify import labelling_2msk


    main_loop_counter = 0

    ### buffering the iterative results
    sliding_mask = np.copy(binary_mask)

    labels = np.array([None for l in locations])
    idv_mask_list = [None for l in locations]
    loc_list = [] 

    prev_locations = np.array([])
    prev_idv_mask = []
    prev_labels = np.array([])

    loc_needs_refinement = [True for l in locations]
    loc_has_converged = [False for l in locations]


    while True:
        main_loop_counter += 1
        print('\nMain consistency loop: ', main_loop_counter)

        # ----------------------------- checking stability ---------------------------------
        bool_stable_locs = check_locs_updated(prev_locations, locations)

        bool_stable_idv_masks = check_masks_updated(prev_idv_mask, idv_mask_list)

        bool_stable_labels = check_labels_updated(prev_labels, labels)

        print('if locations coherent: {}. \nif masks coherent: {}. \nif labels coherent: {}. \n'.format(
                                        bool_stable_locs, bool_stable_idv_masks, bool_stable_labels))

        if bool_stable_locs and bool_stable_idv_masks and bool_stable_labels:
            print('Success coherence reached. ')
            break 

        else:
            print(' previous locations: ', prev_locations)
            print(' current locations: ', locations)

            print(' previous labels: ', prev_labels)
            print(' current labels: ', labels)

        if main_loop_counter >= 10:
            print('Maximum iteration reached, breaking the main loop. \n')
            break 

        # ---------------------------- location - segmentation refinement ---------------------

        prev_locations = np.copy(locations)
        prev_idv_mask = idv_mask_list.copy()
        prev_labels = np.copy(labels)

        loc_list = [loc for loc in locations]

        if len(loc_list) > 0:

            locations, idv_mask_list = locations_refiner_iter(loc_list, idv_mask_list, labels, pir_img, sliding_mask,
                                                              loc_needs_refinement, loc_has_converged,
                                                              model_file_seg_idv, seg_idv_norm, max_iter=10)
            time_dup_start = time.time()
            locations, idv_mask_list, loc_has_converged, loc_needs_refinement = locations_and_masks_without_duplication(locations, idv_mask_list,
                                                                                                        loc_has_converged, loc_needs_refinement)
            time_dup_end = time.time()
            print("\nduplication time:", time_dup_end-time_dup_start,"\n")

            individual_binary_mask = individual_binary_mask_from_list(idv_mask_list)

            binary_mask = np.logical_or(sliding_mask, individual_binary_mask).astype(np.int)
            time_label_start = time.time()
            labels = labelling_2msk(sliding_mask, individual_binary_mask, locations, loc_has_converged, 
                                    model_file_id_group, model_file_id_cer, model_file_id_thor, model_file_id_lum)
            time_label_end = time.time()
            print("\nlabelling time:", time_label_end - time_label_start,"\n")

        else:
            individual_binary_mask = np.zeros(binary_mask.shape)


        extra_locations = []
        extra_labels = []

        # -----------------1. missing locations from residual connected components------------------------

        cc_counter = 0
        num_locations_list = [len(locations), ]

        while True:
            cc_counter += 1
            print(' == residual connected components, loop {}== '.format(cc_counter))

            if len(num_locations_list) > 2:
                for i in range(2, len(num_locations_list)+1):
                    if num_locations_list[-i] == num_locations_list[-1]:
                        print(' Oscillation detected in cc loop in round {}. '.format(i-1))
                        break 

            if cc_counter >= 20:
                print('Maximum iteration reached, breaking the cc loop. \n')
                break 

            ## compute the connected components of masks residual
            diff = binary_mask - individual_binary_mask
            diff[diff!=1] = 0

            ## erode the residual, avoiding components connected by noise
            if len(locations) < 5:
                from scipy.ndimage import morphology
                diff = morphology.binary_erosion(diff).astype(diff.dtype)

            time_cc_start = time.time()
            ccomponents, cc_labels = filtered_connected_components(diff, locations, labels, idv_mask_list)
            time_cc_end = time.time()
            print("\ncc time:",time_cc_end - time_cc_start,"\n")
            if not ccomponents:
                print('No more connected components found.')
                break 

            print(' found {} connected components. '.format(len(ccomponents)))
            cc_locations, cc_labels = get_extra_locations_from_components(ccomponents, cc_labels, locations)

            extra_locations = extra_locations + cc_locations
            extra_labels = extra_labels + cc_labels

            extra_locations = np.array(extra_locations)
            print(' {} extra locations from connected components. '.format(len(extra_locations)))

            extra_masks, extra_locs, extra_refinement, extra_convergence = extra_locations_refinement(extra_locations, 
                                                                                                      locations,
                                                                                                      extra_labels, 
                                                                                                      pir_img, 
                                                                                                      binary_mask,
                                                                                                      idv_mask_list,
                                                                                                      model_file_seg_idv,
                                                                                                      seg_idv_norm)            

            extra_locations = []
            extra_labels = [] 

            if not extra_masks:
                print('No extra masks added.')
                break 

            locations, labels, idv_mask_list, individual_binary_mask, binary_mask, loc_needs_refinement, loc_has_converged = update_locations_masks_labels(extra_masks, 
                                                                                            extra_locs, extra_refinement, extra_convergence,
                                                                            locations, idv_mask_list, loc_needs_refinement, loc_has_converged,
                                                                                sliding_mask, binary_mask, model_file_id_group, model_file_id_cer,
                                                                                    model_file_id_thor, model_file_id_lum)            


            num_locations_list.append(len(locations))


        # ------------------------- 2. missing locations from gaps -----------------------------------

        gap_counter = 0
        while True:
            
            gap_counter += 1
            print(' == gap fill, loop {}== '.format(gap_counter))

            if gap_counter >= 10:
                break


            extra_locations_from_gap, extra_labels_from_gap = get_extra_locations_from_intervals(locations, labels)

            if len(extra_locations_from_gap) == 0:
                break 

            extra_locations = extra_locations + extra_locations_from_gap
            extra_labels = extra_labels + extra_labels_from_gap

            extra_locations = np.array(extra_locations)

            extra_masks, extra_locs, extra_refinement, extra_convergence = extra_locations_refinement(extra_locations, 
                                                                                                      locations,
                                                                                                      extra_labels, 
                                                                                                      pir_img, 
                                                                                                      None,
                                                                                                      idv_mask_list,
                                                                                                      model_file_seg_idv,
                                                                                                      seg_idv_norm)            
            extra_locations = []
            extra_labels = [] 
            
            if not extra_masks:
                print('No extra masks added.')
                break 

            locations, labels, idv_mask_list, individual_binary_mask, binary_mask, loc_needs_refinement, loc_has_converged = update_locations_masks_labels(extra_masks, 
                                                                                            extra_locs, extra_refinement, extra_convergence,
                                                                            locations, idv_mask_list, loc_needs_refinement, loc_has_converged,
                                                                                sliding_mask, binary_mask, model_file_id_group, model_file_id_cer,
                                                                                    model_file_id_thor, model_file_id_lum)            

        # ------------------------- 3. missing locations from fishing -----------------------------------

        #### fishing down 
        fish_down_counter = 0
        while True:
            
            fish_down_counter += 1
            print(' == fishing down, loop {} == '.format(fish_down_counter))

            if fish_down_counter >= 5:
                break 

            if labels[-1] < 25:
                extra_locations_fished = fish_down(locations[-2], locations[-1], pir_img.shape)
                if extra_locations_fished is not None:
                    extra_locations.append(extra_locations_fished)
                    extra_labels.append(labels[-1]+1) 
                    print('add fished loc and label {}'.format(labels[-1]+1))

            if len(extra_locations) == 0:
                break 

            extra_locations = np.array(extra_locations)

            extra_masks, extra_locs, extra_refinement, extra_convergence = extra_locations_refinement(extra_locations, 
                                                                                                      locations,
                                                                                                      extra_labels, 
                                                                                                      pir_img, 
                                                                                                      None,
                                                                                                      idv_mask_list,
                                                                                                      model_file_seg_idv,
                                                                                                      seg_idv_norm,
                                                                                                      fishing=True)            
            extra_locations = []
            extra_labels = [] 
            
            if not extra_masks:
                print('No extra masks added.')
                break 

            locations, labels, idv_mask_list, individual_binary_mask, binary_mask, loc_needs_refinement, loc_has_converged = update_locations_masks_labels(extra_masks, 
                                                                                            extra_locs, extra_refinement, extra_convergence,
                                                                            locations, idv_mask_list, loc_needs_refinement, loc_has_converged,
                                                                                sliding_mask, binary_mask, model_file_id_group, model_file_id_cer,
                                                                                    model_file_id_thor, model_file_id_lum)            

        #### fishing up 
        fish_up_counter = 0
        while True:
            
            fish_up_counter += 1
            print(' == fishing up, loop {} == '.format(fish_up_counter)) 

            if fish_up_counter >= 5:
                break 


            if labels[0] > 1:
                extra_locations_fished = fish_up(locations[0], locations[1])
                if extra_locations_fished is not None:
                    extra_locations.append(extra_locations_fished)
                    extra_labels.append(labels[0]-1) 
                    print('add fished loc and label {}'.format(labels[0]-1))

            if len(extra_locations) == 0:
                break

            extra_locations = np.array(extra_locations)


            extra_masks, extra_locs, extra_refinement, extra_convergence = extra_locations_refinement(extra_locations, 
                                                                                                      locations,
                                                                                                      extra_labels, 
                                                                                                      pir_img, 
                                                                                                      None,
                                                                                                      idv_mask_list,
                                                                                                      model_file_seg_idv,
                                                                                                      seg_idv_norm,
                                                                                                      fishing=True)            
            extra_locations = []
            extra_labels = [] 
            
            if not extra_masks:
                print('No extra masks added.')
                break 

            locations, labels, idv_mask_list, individual_binary_mask, binary_mask, loc_needs_refinement, loc_has_converged = update_locations_masks_labels(extra_masks, 
                                                                                            extra_locs, extra_refinement, extra_convergence,
                                                                            locations, idv_mask_list, loc_needs_refinement, loc_has_converged,
                                                                                sliding_mask, binary_mask, model_file_id_group, model_file_id_cer,
                                                                                    model_file_id_thor, model_file_id_lum)            

        # ------------------------------------------------------------------------------------------------

    multi_label_mask = aggregate_multi_label_segmentation(idv_mask_list, labels)

    assert locations.all()

    return multi_label_mask, locations, labels, loc_has_converged