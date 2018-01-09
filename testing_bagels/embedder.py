import numpy as np
from numpy import linalg as ln

def biggest_eigenvalue(m1):
    return(abs(calculate_eigenvalues(m1)[0]))

def calculate_average_degree(m1):
    degrees = []
    for i in m1:
        non_zero = np.count_nonzero(i)
        degrees.append(non_zero)
    return(np.mean(degrees))

def calculate_average_distances(m1):
    distances = []
    for i in m1:
        distance = np.sum(i)
        distances.append(distance)
    return(np.mean(distances))

def calculate_biggest_eigengap(m1):
    eigs = calculate_eigenvalues(m1)
    maximum = 0
    for i in range(len(m1) - 1):
        gap = eigs[i+1] - eigs[i]
        if gap > maximum:
            maximum = gap
    return(abs(maximum))

def calculate_eigenvalues(m1):
    eigenvalues = ln.eig(m1)[0]
    return(eigenvalues)

def smallest_eigenvalue(m1):
    return(abs(min(calculate_eigenvalues(m1))))

def sum_eigenvalues(m1):
    return(abs(sum(calculate_eigenvalues(m1))))

def embed(m1):
    vec = [calculate_average_distances(m1),
     calculate_biggest_eigengap(m1), sum_eigenvalues(m1)]
    return(vec)
