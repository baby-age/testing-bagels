import numpy as np
import pandas as pand
import scipy.io
import os
import csv

def read_neuro_scores(csv_path):
    neuro_scores = {}

    with open(csv_path) as csvfile:
        rdr = csv.reader(csvfile, delimiter=";")

        next(rdr)
        for row in rdr:
            baby_id, c1, c2 = row[0], float(row[2]), float(row[3])
            neuro_scores[baby_id] = (c1, c2)


    return neuro_scores

def read_data(path, group, modality, frequency_range, csv_path):
    params = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3,
              'control': 'CONT', 'preterm': 'KEKE', 'PPC': 'dbPLI',
              'AAC': 'oCC', 'active sleep': 'AS_A_dbpli.mat', 'quiet sleep': 'TA_A_dbpli.mat'}

    files = sorted(list(filter(lambda x: x.startswith(params['preterm']), os.listdir(path))))

    neuro_scores = read_neuro_scores(csv_path)

    data = []
    for i in range(len(files)):
        graph = read_graph(path + files[i] + '/' + params[modality] + '/' + params['active sleep'], params[frequency_range])

        try:
            # Change the last number to 1 to get the latter score
            neuro_score = neuro_scores["'" +files[i]+ "'"][0]
            data.append((graph, neuro_score))
        except KeyError:
            pass

    df = pand.DataFrame(data, columns = ['X', 'y'])
    return(df)

def read_graph(location, frequency_range):
    mat = scipy.io.loadmat(location)['dbPLI']
    theta_range_matrix = mat[0][frequency_range]

    return theta_range_matrix
