__copyright__ = \
"""
Copyright &copyright © (c) 2021 Inria Grenoble Rhône-Alpes.
All rights reserved.

This source code is to be used for academic research purposes only.
For commercial uses of the code, please send an email to edmond.boyer@inria.fr and sergi.pujades@inria.fr

"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Di Meng"


def load_classifier(model_file, classify_level, with_centermap):

    from SpineSeg.classifier import encoder3d
    import torch 

    if classify_level == 'group':
        nclass = 3
    elif classify_level == 'cervical':
        nclass = 7
    elif classify_level == 'thoracic':
        nclass = 12
    elif classify_level == 'lumbar':
        nclass = 5
    else:
        raise ValueError('wrong classify level.')

    if with_centermap:
        model = encoder3d(n_classes=nclass, in_channel=2)
    else:
        model = encoder3d(n_classes=nclass, in_channel=1)

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to(torch.device("cuda"))
    model.eval()

    return model


def toTensor(x, with_centermap):
    import torch

    assert len(x.shape) == 3

    x = torch.from_numpy(x).to(dtype=torch.float32)
    out = x.view(1, 1, x.size(0), x.size(1), x.size(2))

    if with_centermap:
        from classifier_test_public_set import create_centermap
        centermap = create_centermap()
        centermap = torch.from_numpy(centermap).to(dtype=torch.float32)
        centermap = centermap.view(1, 1, centermap.size(0), centermap.size(1), centermap.size(2))
        out = torch.cat((out, centermap), 1)

    if torch.cuda.is_available(): out = out.cuda()

    return out


def apply_classifier(msk_cube, model_file, classify_level, with_centermap):
    import torch
    import numpy as np 

    model = load_classifier(model_file, classify_level, with_centermap)

    msk_cube = toTensor(msk_cube, with_centermap)
    pred = model(msk_cube)

    pred_label = torch.argmax(pred)
    pred_prob = torch.nn.functional.softmax(pred, dim=1)

    pred_label = pred_label.cpu().detach().numpy().astype(np.int)
    pred_prob = pred_prob.cpu().detach().numpy().astype(np.float32)

    return pred_label, pred_prob


def centroid_translate(msk, cube_size):
    import numpy as np 

    msk_copy = np.copy(msk)

    cube_size_x, cube_size_y, cube_size_z = cube_size[:]

    x_trans = cube_size_x//2
    y_trans = cube_size_y//2
    z_trans = cube_size_z//2

    msk_copy = np.pad(msk_copy, [(cube_size_x//2, cube_size_x//2), (cube_size_y//2, cube_size_y//2), (cube_size_z//2, cube_size_z//2)], mode='constant', constant_values=0)

    return msk_copy, x_trans, y_trans, z_trans


def extract_single_cube(loc, msk):

    import numpy as np 

    cube_size = (128, 128, 128)
    cube_size_x, cube_size_y, cube_size_z = cube_size[:]

    msk_pad, x_trans, y_trans, z_trans = centroid_translate(msk, cube_size)

    x,y,z = loc[:]
    x,y,z = int(round(x)+x_trans), int(round(y)+y_trans), int(round(z)+z_trans)

    msk_cube = msk_pad[x-cube_size_x//2:x+cube_size_x//2, y-cube_size_y//2:y+cube_size_y//2, \
                        z-cube_size_z//2:z+cube_size_z//2]

    return msk_cube


def pred_label_to_spesific_label(pred_label, classify_level):

    if classify_level == 'group':
        pred_label = pred_label

    elif classify_level == 'cervical':
        pred_label = pred_label + 1

    elif classify_level == 'thoracic':
        pred_label = pred_label + 8

    elif classify_level == 'lumbar':
        pred_label = pred_label + 20

    else:
        raise ValueError('wrong classify level.')

    return pred_label


def per_vertebra_classification(msk_cube, model_file, classify_level, with_centermap=False):

    pred_label, pred_prob = apply_classifier(msk_cube, model_file, classify_level, with_centermap)
    pred_label = pred_label_to_spesific_label(pred_label, classify_level)


    return pred_label, pred_prob



def predict_individuals(vol, locations, loc_has_converged,
                        group_model_file, cer_model_file, 
                        thor_model_file, lum_model_file):
    import numpy as np 

    n_verts = len(locations)

    weight_group_cost = 5
    c1c2_weight = 5

    probabilities = np.zeros((n_verts, 24))

    group_cost = np.ones((n_verts, 24))*weight_group_cost
    pred_labels = []

    # input volume is binary
    for i, (loc, has_conv) in enumerate(zip(locations, loc_has_converged)):

        if not has_conv:
            probabilities[i,:] = np.ones(24) * (1./24.)
            pred_labels.append(-1)
            continue

        vert_cube = extract_single_cube(loc, vol)

        # predict the group level
        pred_group, _ = per_vertebra_classification(vert_cube, group_model_file, 'group')

        if pred_group == 0:
            pred_idv, pred_prob = per_vertebra_classification(vert_cube, cer_model_file, 'cervical')

            probabilities[i,0:7] = pred_prob
            group_cost[i,0:7] = 0

            if pred_idv == 1:
                group_cost[i,0] = -c1c2_weight
            elif pred_idv == 2:
                group_cost[i,1] = -c1c2_weight
            else:
                pass

        elif pred_group == 1:
            pred_idv, pred_prob = per_vertebra_classification(vert_cube, thor_model_file, 'thoracic')
            probabilities[i,7:19] = pred_prob
            group_cost[i, 7:19] = 0     

        elif pred_group == 2:
            pred_idv, pred_prob = per_vertebra_classification(vert_cube, lum_model_file, 'lumbar')
            probabilities[i,19:] = pred_prob
            group_cost[i,19:] = 0              

        else:
            raise ValueError('wrong group prediction.')

        pred_labels.append(pred_idv)
    

    return probabilities, pred_labels, group_cost


def getPath(probabilities, g_cost):
    from SpineSeg import graph
    import networkx as nx

    G = nx.DiGraph()
    num_bones = probabilities.shape[0]
    num_vert = probabilities.shape[1]
    name_to_index = dict()
    
    
    nodes = []
    nodes = graph.create_nodes(G, num_bones, num_vert, nodes, name_to_index)
    graph.create_edges(G, num_bones, num_vert, nodes, probabilities, g_cost, name_to_index)

    
    path = graph.get_shortestPath(G, probabilities)
    optimized_path = graph.path_from_names(G, path)
    optimized_path = graph.relabel(optimized_path)

    # avoid the situation of both "absence of T12" and "existence of L6"
    if (18 in optimized_path and 20 in optimized_path and 25 in optimized_path) and 19 not in optimized_path:
        optimized_path = graph.relabel_T12_L6(optimized_path)


    return optimized_path


def labelling(vol, locations, loc_has_converged, 
              group_model_file, cer_model_file, 
              thor_model_file, lum_model_file):

    probabilities, pred_labels, group_cost = predict_individuals(vol, locations, loc_has_converged, group_model_file, cer_model_file, thor_model_file, lum_model_file)
    path = getPath(probabilities, group_cost)

    return path 


def labelling_2msk(mask_bin, mask_indiv, 
                   locations, loc_has_converged,
                   group_model_file, cer_model_file, 
                   thor_model_file, lum_model_file):

    import numpy as np 
    import torch

    probabilities_bin, pred_labels, g_cost_bin = predict_individuals(mask_bin, locations, loc_has_converged, group_model_file, cer_model_file, thor_model_file, lum_model_file)
    probabilities_ind, pred_labels, g_cost_ind = predict_individuals(mask_indiv, locations, loc_has_converged, group_model_file, cer_model_file, thor_model_file, lum_model_file)


    probabilities = np.mean(np.array([probabilities_bin, probabilities_ind]), axis=0)
    g_cost = np.mean(np.array([g_cost_bin, g_cost_ind]), axis=0)

    path = getPath(probabilities, g_cost) 

    return path 


