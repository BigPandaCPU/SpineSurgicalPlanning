__copyright__ = \
"""
Copyright &copyright © (c) 2021 Inria Grenoble Rhône-Alpes.
All rights reserved.

This source code is to be used for academic research purposes only.
For commercial uses of the code, please send an email to edmond.boyer@inria.fr and sergi.pujades@inria.fr

"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Eslam Mohammed, Di Meng"



import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import networkx.drawing.nx_pylab as nx_plot



plt.rcParams.update({'font.size': 16})
plt.rc('axes', titlesize=24)
cervicals = [1, 2, 3, 4, 5, 6, 7]
thoracics = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
lumbars = [20, 21, 22, 23, 24]
class_names = [*cervicals, *thoracics, *lumbars]


BINARY_MAX_WEIGHT = 10e8

### for verse
WEIGHT_28 = 2.0
WEIGHT_25 = 1
WEIGHT_NO19 = 3.0  

### for other dataset 
# WEIGHT_28 = 10e8
# WEIGHT_25 = 10e8
# WEIGHT_NO19 = 10e8  

def name_from_indices(i, j):
    return"{:2d}_{:2d}".format(i,j)

def indices_from_name(name):
    return int(name.split("_")[0]), int(name.split("_")[1])

def get_key_from_value(dict_, val):
    for (key, value) in dict_.items():
        if value == val:
            return key

        
def create_nodes(graph, N, V, study, name_to_index):

    graph.add_nodes_from(['src', 'dst'])  
    for i in range(N):
        for j in range(V):
            name = name_from_indices(i,j)
            study.append(name)
            name_to_index[name] = V * i + j
            i_n, j_n = indices_from_name(name)
            assert(i_n == i and j_n == j)
    
    graph.add_nodes_from(study)


def create_edges(graph, N, V, study, prob, g_cost, name_to_index):

    for name in list(graph):
        if name == 'src':

            for j in range(V):
                node = name_from_indices(0,j)
                probability_of_node = prob[0][j]
                cost_of_node = g_cost[0][j]
                unary_term = 1 - probability_of_node + cost_of_node
                graph.add_edge('src', node, weight= unary_term)

        else:  
            if name == 'dst':
                continue    
            index_node = name_to_index[name]
            index_node_key = get_key_from_value(name_to_index, index_node)
            i_n, j_n = indices_from_name(name)
            if i_n == N-1:
                # break
                graph.add_edge(index_node_key, 'dst', weight= BINARY_MAX_WEIGHT)
            else:
                for j in range(V):
                    next_node_index = V * (i_n + 1) + j
                    next_node_key = get_key_from_value(name_to_index, next_node_index)
                    probability_of_node = prob[i_n+1][j]
                    cost_of_node = g_cost[i_n+1][j]
                    unary_term = 1 - probability_of_node + cost_of_node  #uniary term (probability)
                    
                    if j_n + 1 == j:
                        weight = unary_term 
                    elif j_n - 1 == j:                  # Skip the connection of the same label 
                        continue 
                    elif j == j_n == 23:                # Case of L6 (by handling 2 consecutives L5)
                        weight = unary_term + WEIGHT_25
                    elif j_n == 17 and j == 19:         # Case of missing T12 (establish a connection between T11 and L1)
                        weight = unary_term + WEIGHT_NO19
                    elif j_n == j == 18:                # Case of T13 (by handling 2 consecutives T12)
                        weight = unary_term + WEIGHT_28
                    elif j_n == 23 and j == 18:
                        weight = unary_term + WEIGHT_28
                    elif j_n == j:                      # Skip the connection of the label before
                        continue
                    else:
                        weight = unary_term + BINARY_MAX_WEIGHT
                            
                    graph.add_edge(index_node_key, next_node_key, weight= weight)
                    

def plot_graph(graph):
    
    plt.figure(figsize=(20, 10))
    nx.draw_circular(graph, with_labels=True, font_weight='bold')
    plt.show()
    

def get_shortestPath(graph, prob):

    path = nx.dijkstra_path(graph, 'src', 'dst', weight='weight')
    path = path[1:-1]

    return path


def path_from_predictions(volume):
    path_map = class_names * volume.shape[0]
    path = []
    preds = np.argmax(volume, axis=1)
    for i, x in enumerate(np.nditer(preds)):
        path.append(path_map[x + 24*i])
        
    return path

def path_from_names(graph, path):
    idx = []                                                 # For storing the indices of the vertices
    for i in path:
        idx.append(list(graph).index(i))

    bone_names = class_names * len(path)
    path_names = [bone_names[i-2] for i in idx]

    return path_names

def relabel(corrections):
    for i in range(1, len(corrections)):
        
        if corrections[i] == 24 and corrections[i-1] == 24:
            corrections[i] = 25
            
        elif corrections[i] == 19 and corrections[i-1] == 19:

            if i == len(corrections)-1:
                corrections[i] = 20
            else:
                corrections[i] = 28
        else:
            pass
    return corrections 


def relabel_T12_L6(labels):

    for i, l in enumerate(labels):
        if l > 19 and l != 28:
            labels[i] -= 1

    return labels