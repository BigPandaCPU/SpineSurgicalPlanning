__copyright__ = \
"""
Copyright &copyright © (c) 2021 Inria Grenoble Rhône-Alpes.
All rights reserved.

This source code is to be used for academic research purposes only.
For commercial uses of the code, please send an email to edmond.boyer@inria.fr and sergi.pujades@inria.fr

"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Di Meng"



def regressor_gap_two_sides(pre_gap, next_gap, level=None):
    # predict gap from previous and next gaps 

    if level == 1:   # cervical 
        a = 0.55126
        b = 0.45999
        c = -0.08915

    elif level == 2: # thoracic
        a = 0.57909
        b = 0.44036
        c = -0.24396

    elif level == 3: # lumbar 
        a = 0.56488
        b = 0.46991
        c = -0.73769

    else:            # generic
        a = 0.56909
        b = 0.45370
        c = -0.36618

    return a*pre_gap + b*next_gap + c 

def regressor_gap_one_side(neighbor_gap, direc, level=None):
    # predict gap from either previous or next gap 

    if direc == 'pre':
        if level == 1:
            a = 0.92428
            c = 2.40965
        elif level == 2:
            a = 0.93802
            c = 2.29442
        elif level == 3:
            a = 0.9533
            c = 1.9698
        else:
            a = 0.942422
            c = 2.170071

    elif direc == 'next':
        if level == 1:
            a = 0.98403
            c = -0.13289
        elif level == 2:
            a = 0.97728
            c = -0.07831
        elif level == 3:
            a = 0.96615
            c = 0.23980
        else:
            a = 0.974222
            c = 0.025959

    else:
        raise NotImplementedError('Unknown direction {}. Options: pre | next.'.format(direc))

    return a*neighbor_gap + c 


def regressor_size_one_side(neighbor_size, direc, level=None):
    # predict the size using previous or next size

    if direc == 'pre':
        if level == 1:
            a = 1.033e+00
            c = 1.471e+03
        elif level == 2:
            a = 1.034e+00
            c = 1.354e+03
        elif level == 3:
            a = 1.052e+00 
            c = 9.188e+02
        else:
            a = 1.039
            c = 1.272e+03

    elif direc == 'next':
        if level == 1:
            a = 0.9235
            c = 497.0120
        elif level == 2:
            a = 0.93555
            c = -140.92364
        elif level == 3:
            a = 0.93635
            c = -269.94402
        else:
            a = 0.933510
            c = -54.083502

    else:
        raise NotImplementedError('Unknown direction {}. Options: pre | next.'.format(direc))

    return a*neighbor_size + c 