import numpy as np
import pandas as pand
import scipy.io
import os


    df = pand.DataFrame(data, columns = ['X', 'y'])
    return(df)

def read_graph(location, frequency_range):
    mat = scipy.io.loadmat(location)['dbPLI']
    theta_range_matrix = mat[0][frequency_range]

    return theta_range_matrix, 0
