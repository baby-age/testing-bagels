import numpy as np
import os
import scipy.io
import pandas as pd

def preprocess_whole_data(data):
    preprocessed_data = pd.DataFrame()
    preprocessed_data['X'] = [preprocess_matrix(x) for x in data['X']]
    preprocessed_data['y'] = data['y']
    return preprocessed_data
    
def preprocess_matrix(m1):
    m1 = abs(m1)
    m1 = get_symmetric_matrix(m1)
    m1 = correct_with_fidelity_operator(m1)
    m1 = normalize_matrix(m1)
    return m1

def get_symmetric_matrix(m1):
    A = np.maximum(m1, m1.transpose())
    return A

def correct_with_fidelity_operator(m1):
    fidelity_operator = get_fidelity_operator()
    A = np.dot(m1, fidelity_operator)
    return A

def get_fidelity_operator():
    # Set path to the location of FidelityOperator_n58.mat on your computer
    path = '/home/matleino/testing-bagels/FidelityOperator_n58.mat'
    fidelity_operator_matrix = scipy.io.loadmat(path)['FidelityOperator']
    return fidelity_operator_matrix

def normalize_matrix(m1):
    return np.divide(m1, sum(sum(m1)))