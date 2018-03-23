import scipy.io
import os
import unicodedata
import pandas as pand
import testing_bagels.graphread as read


def get_brain_areas(path):
    mat = scipy.io.loadmat(path)
    areas = mat['MyAtlas'][0][0][3]
    areas_dict = {'C': [], 'T': [], 'F': [], 'O': []}

    for i in range(len(areas)):
        area_code = areas[i][0][0]
        areas_dict[area_code].append(i + 1)

    return areas_dict

def get_intra_areal_connectivity(matrix, area_nodes_list):
    intra_areal_connectivity_sum = 0
    count = 0
    for i in range(len(area_nodes_list)):
        rowcount = 0
        for j in range(i + 1, len(area_nodes_list)):
            intra_areal_connectivity_sum += matrix[area_nodes_list[i] - 1][area_nodes_list[j] - 1]
            count += 1

    intra_areal_connectivity = intra_areal_connectivity_sum / count
    return intra_areal_connectivity

def get_inter_areal_connectivity(matrix, area_nodes_list_1, area_nodes_list_2):
    inter_areal_connectivity_sum = 0
    count = 0
    rowcount = 0
    for i in range(len(area_nodes_list_1)):
        for j in range(len(area_nodes_list_2)):
            inter_areal_connectivity_sum += matrix[area_nodes_list_1[i] - 1][area_nodes_list_2[j] - 1]
            count += 1

    inter_areal_connectivity = inter_areal_connectivity_sum / count
    return inter_areal_connectivity

def get_matrix_connections(matrix, areas_dict):
    frontal_connections = get_intra_areal_connectivity(matrix, areas_dict['F'])
    central_connections = get_intra_areal_connectivity(matrix, areas_dict['C'])
    occipital_connections = get_intra_areal_connectivity(matrix, areas_dict['O'])
    temporal_connections = get_intra_areal_connectivity(matrix, areas_dict['T'])

    fc_connections = get_inter_areal_connectivity(matrix, areas_dict['F'], areas_dict['C'])
    fo_connections = get_inter_areal_connectivity(matrix, areas_dict['F'], areas_dict['O'])
    ft_connections = get_inter_areal_connectivity(matrix, areas_dict['F'], areas_dict['T'])
    co_connections = get_inter_areal_connectivity(matrix, areas_dict['C'], areas_dict['O'])
    ct_connections = get_inter_areal_connectivity(matrix, areas_dict['C'], areas_dict['T'])
    ot_connections = get_inter_areal_connectivity(matrix, areas_dict['O'], areas_dict['T'])

    return (frontal_connections,
    central_connections,
    occipital_connections,
    temporal_connections,
    fc_connections,
    fo_connections,
    ft_connections,
    co_connections,
    ct_connections,
    ot_connections)


"""
This is the main function of this class.
Parameter data should be the output of read_data-function from graphread.py-class
Parameter path_to_MyAtlas_n58 should be the path to the file MyAtlas_n58.mat for example 
'/home/matleino/tutkijalinja/show_brain areas/MyAtlas_n58.mat'
Output is a pandas-dataframe with dimensions 33x10 where each row is for one graph and each column is the average of connections
in one area of the brain or average connections between two brain areas.
"""
def get_mean_strength_of_connections(data, path_to_MyAtlas_n58):
    areas_dict = get_brain_areas(path_to_MyAtlas_n58)
    connections = []

    for i in range(len(data['X'])):
        matrix = data['X'][i]
        connections.append(get_matrix_connections(matrix, areas_dict))

    df = pand.DataFrame(connections, columns = ['Frontal_connections', 'Central_connections', "Occipital_connections", "Temporal_connections", 
                                        "FC_connections", "FO_connections", "FT_connections", "CO_connections", "CT_connections", "OT_connections"])
    
    return df