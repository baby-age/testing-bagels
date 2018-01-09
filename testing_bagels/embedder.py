import numpy as np
from numpy import linalg as ln

def biggest_eigenvalue(m1):
    return(calculate_eigenvalues(m1)[0])

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

def calculate_eigenvalues(m1):
    eigenvalues = ln.eig(m1)[0]
    return(eigenvalues)

def smallest_eigenvalue(m1):
    return(min(calculate_eigenvalues(m1)))

def sum_eigenvalues(m1):
    return(sum(calculate_eigenvalues(m1)))

def embed(m1):
    vec = [calculate_average_degree(m1), calculate_average_distances(m1), sum_eigenvalues(m1), biggest_eigenvalue(m1), smallest_eigenvalue(m1)]
    return(vec)


A = np.matrix("1 1 0 0; 1 1 1 0; 0 1 1 1; 0 0 1 1")

print(embed(A))
