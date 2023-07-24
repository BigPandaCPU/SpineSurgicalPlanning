

def get_class_weight(group_level):
    import numpy as np 
    from .utils import read_json_file

    vert_count = read_json_file('classifier_num_of_each_label.json')

    if group_level == 'group':
        arr = np.zeros(3)
        for l in range(1, 8):
            arr[0] += vert_count[str(l)]
        for l in range(8,20):
            arr[1] += vert_count[str(l)]
        for l in range(20, 25):
            arr[2] += vert_count[str(l)]
        return 1.0/arr 

    elif group_level == 'cervical':
        arr = np.zeros(7)
        for l in range(1, 8):
            arr[l-1] += vert_count[str(l)]
        return 1.0/arr

    elif group_level == 'thoracic':
        arr = np.zeros(12)
        for l in range(8, 20):
            arr[l-8] += vert_count[str(l)]
        return 1.0/arr

    elif group_level == 'lumbar':
        arr = np.zeros(5)
        for l in range(20, 25):
            arr[l-20] += vert_count[str(l)]
        return 1.0/arr 

    else:
        raise NotImplementedError('Unknown group_level [{}] to compute class weights.'.format(group_level))



if __name__ == "__main__":
    import argparse, os
    from data_generator import read_annotation
    from .utils import write_dict_to_file

    parser = argparse.ArgumentParser("Compute class weight: data distribution of cervical, thoracic and lumbar")

    parser.add_argument('-D', '--dataset_dir', type=str, help='path to the verse20 training set')
    parser.add_argument('-L', '--train_ID_list', nargs='+', help='a list of scan IDs, eg. verse008, GL003, ...')
    parser.add_argument('-S', '--save_dir', type=str, default='/morpheo-nas2/dmeng/verse20/models_classifier')
    args = parser.parse_args()
    
    vert_count = dict()
    for label in range(1, 25):
        vert_count[str(label)] = 0

    for ID in args.train_ID_list:
        anno = read_annotation(ID, args.dataset_dir)
        labels = anno['labels']

        for label in labels:
            label = 19 if label == 28 else label 
            label = 24 if label == 25 else label 
            vert_count[str(label)] += 1

    write_dict_to_file(vert_count, 'classifier_num_of_each_label.json')