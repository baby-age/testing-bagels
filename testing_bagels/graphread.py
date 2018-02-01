import numpy as np
import pandas as pand
import scipy.io
import os

def read_data(path, group, modality, frequency_range):
    params = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3,
              'control': 'CONT', 'preterm': 'KEKE', 'PPC': 'dbPLI',
              'AAC': 'oCC', 'active sleep': 'AS_A_dbpli.mat', 'quiet sleep': 'TA_A_dbpli.mat'}

    files = sorted(list(filter(lambda x: x.startswith(params['preterm']), os.listdir(path))))
    data = []
    for i in range(len(files)):
        data.append(read_graph(path + files[i] + '/' + params[modality] + '/' + params['active sleep'], params[frequency_range]))

    df = pand.DataFrame(data, columns = ['X', 'y'])
    return(df)

def read_graph(location, frequency_range):
    mat = scipy.io.loadmat(location)['dbPLI']
    theta_range_matrix = mat[0][frequency_range]

    return theta_range_matrix, 0
