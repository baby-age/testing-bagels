import numpy as np
import pandas as pand
import scipy.io
import os
import csv

def read_neuro_scores(csv_path):
    neuro_scores = {}
    with open(csv_path) as csvfile:
        rdr = csv.reader(csvfile, delimiter=",")
        next(rdr)
        for row in rdr:
            baby_id, c1, c2 = row[0], float(row[2]), float(row[3])
            neuro_scores[baby_id] = (c1, c2)
    
    return neuro_scores

def read_control_neuro_scores(csv_path):
    control_labels = scipy.io.loadmat(csv_path)['NeuroScores']
    healthy_babies = control_labels[0][0][0]
    healthy_c1 = control_labels[0][0][1]
    healthy_c2 = control_labels[0][0][2]

    neuro_scores = {}
    for i, entry in enumerate(healthy_babies):
        baby_id = "'" + entry[0][0] + "'"
        c1 = healthy_c1[i][0]
        c2 = healthy_c2[i][0]
        neuro_scores[baby_id] = (c1, c2)

    return neuro_scores

def read_data(path, group, modality, frequency_range, csv_path, sleep_mode):
    if group == 'preterm':
        params = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3,
              'control': 'CONT', 'preterm': 'KEKE', 'PPC': 'dbPLI',
              'AAC': 'oCC', 'active sleep': 'AS_A_dbpli.mat', 'quiet sleep': 'TA_A_dbpli.mat'}
    else:
        params = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3,
              'control': 'CONT', 'preterm': 'KEKE', 'PPC': 'dbPLI',
              'AAC': 'oCC', 'active sleep': 'AS_B_dbpli.mat', 'quiet sleep': 'TA_B_dbpli.mat'}

    files = sorted(list(filter(lambda x: x.startswith(params[group]), os.listdir(path))))

    if group == "preterm":
        neuro_scores = read_neuro_scores(csv_path)
    elif group == "control":
        neuro_scores = read_control_neuro_scores(csv_path)

    data = []
    for i in range(len(files)):
        graph = read_graph(path + files[i] + '/' + params[modality] + '/' + params[sleep_mode], params[frequency_range])
        if len(graph) == 0:
            continue
        try:
            # Change the last number to 1 to get the latter score
            neuro_score = neuro_scores["'" +files[i]+ "'"][1]
            data.append((graph, neuro_score))
        except KeyError:
            pass

    df = pand.DataFrame(data, columns = ['X', 'y'])
    return(df)

def read_graph(location, frequency_range):
    try:
        mat = scipy.io.loadmat(location)['dbPLI']
        theta_range_matrix = mat[0][frequency_range]
        return theta_range_matrix
    except TypeError as error:
        return np.array([])
