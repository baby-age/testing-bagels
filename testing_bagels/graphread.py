import numpy as np
import pandas as pand
import scipy.io
import os

def read_data(path, frequency_range):
    files = sorted(os.listdir(path))    
    data = []
    for i in range(len(files)):
        data.append(read_graph(path + files[i] + '/dbPLI/AS_A_dbpli.mat', frequency_range))

    df = pand.DataFrame(data, columns = ['X', 'y'])
    return(df)

def read_graph(location, frequency_range):
    mat = scipy.io.loadmat(location)['dbPLI']
    theta_range_matrix = mat[0][frequency_range]

    return theta_range_matrix, 0