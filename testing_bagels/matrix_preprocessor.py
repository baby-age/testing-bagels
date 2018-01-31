import numpy as np
import os
import scipy.io


def preprocess_matrix(m1):
    m1 = abs(m1)
    m1 = get_symmetric_matrix(m1)
    m1 = correct_with_fidelity_operator(m1)
    m1 = normalize_matrix(m1)

def get_symmetric_matrix(m1):
    A = np.maximum(A, A.transpose())
    return A

def correct_with_fidelity_operator(m1):
    fidelity_operator = get_fidelity_operator
    A = m1 * fidelity_operator
    return A

def get_fidelity_operator():
location = os.getcwd() + '/FidelityOperator/FidelityOperator_n58.mat'
fidelity_operator_matrix = scipy.io.loadmat(location)['FidelityOperator']
    return fidelity_operator_matrix

def normalize_matrix(m1):
    A = A / sum(sum(A))
    return A