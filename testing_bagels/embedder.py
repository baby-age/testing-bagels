import numpy as np
import graphgen
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
        gap = abs(eigs[i+1] - eigs[i])
        if gap > maximum:
            maximum = gap
    return(maximum)

def calculate_eigenvalues(m1):
    eigenvalues = ln.eig(m1)[0]
    return(eigenvalues)


def eigenmode_volume(m1):
    eigenmodes = ln.eig(m1)
    phi = np.column_stack(eigenmodes[1])
    return(np.dot(np.transpose(phi), calculate_degree_vector(m1)))

def count_adjacents(vec):
    count = 0
    for i in vec:
        if i != 0:
            count = count+1
    return count

def calculate_degree_matrix(m1):
    dim = len(m1)
    degree_matrix = np.diag([count_adjacents(y) for y in m1])
    return(degree_matrix)

def calculate_weighted_degree_matrix(m1):
    dim = len(m1)
    w_degree_matrix = [[sum(x) for x in m1[y]] for y in range(dim)]
    return(w_degree_matrix)

def is_zero(x):
    if x == 0:
        return 0
    else:
        return 1

def calculate_adjacency_matrix(m1):
    adj_m = np.matrix([[is_zero(x) for x in m1[y]] for y in range(len(m1))])
    return(adj_m)

def calculate_laplacian_matrix(m1):
    d = calculate_degree_matrix(m1)
    a = calculate_adjacency_matrix(m1)

    lap_mat = np.subtract(d, a)

    return(lap_mat)

def smallest_eigenvalue(m1):
    return(abs(min(calculate_eigenvalues(m1))))

def sum_eigenvalues(m1):
    return(abs(sum(calculate_eigenvalues(m1))))

def embed(m1):
    vec = [calculate_average_distances(m1),
     calculate_biggest_eigengap(m1), sum_eigenvalues(m1)]
    return(vec)
