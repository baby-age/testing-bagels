import numpy as np
import testing_bagels.graphgen
from numpy import linalg as ln
import itertools

def algebraic_connectivities(m1):
    data_list = list()
    for vector in m1:
        matrix = np.zeros((58,58))
        acc = 0
        for i in range(0, 58):
            for j in range(i+1, 58):
                matrix[i][j] = vector[acc]
                matrix[j][i] = vector[acc]
                acc = acc+1

        lap = calculate_laplacian_matrix(matrix)
        eigenvalues = ln.eig(lap)[0]
        data_list.append(sorted(eigenvalues)[1])

    return(data_list)

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
    eigenmodes = ln.eig(m1)
    return(eigenmodes[0])


def eigenmode_volume(m1):
    eigenmodes = ln.eig(m1)
    phi = np.column_stack(eigenmodes[1])
    return(np.dot(np.transpose(phi), calculate_degree_vector(m1)))

def count_adjacents(vec):
    count = 0

    for i in vec:
        count = sum(i)


    return count



def calculate_weighted_degree_matrix(m1):
    dim = len(m1)
    w_degree_matrix = [[sum(x) for x in m1[y]] for y in range(dim)]
    return(w_degree_matrix)

def if_zero(x):
    if x == 0:
        return 0
    else:
        return 1

def calculate_adjacency_matrix(m1):
    for row in m1:
        for element in row:
            print(element)
    adj_m = np.matrix([[if_zero(x) for x in m1[y]] for y in range(len(m1))])
    return(adj_m)

def calculate_degree_matrix(m1):
    diagonal_degrees = []
    for row in m1:
        diagonal_degrees.append(np.sum(np.sum(row)))
        print(np.sum(np.sum(row)))
    degree_matrix = np.diag(np.diag(diagonal_degrees))

    return(degree_matrix)

"""
def calculate_laplacian_matrix(m1):

    ds = calculate_degree_matrix(m1)
    ads = calculate_adjacency_matrix(m1)
    lap_mat = np.subtract(ds, ads)

    return(lap_mat)
"""
def calculate_laplacian_matrix(m1):
    degree_matrix = np.zeros((58,58))
    for i in range(0, len(m1)):
        degree = sum(m1[i])
        degree_matrix[i,i] = degree

    laplacian_matrix = np.subtract(degree_matrix, m1)

    return(laplacian_matrix)


def normalised_laplacian_matrix(m1):
    d = calculate_degree_matrix(m1)
    lap_mat = calculate_laplacian_matrix(m1)

    d_minus = np.sqrt(ln.inv(d))
    normalised_lap_mat = np.dot(d_minus, np.dot(lap_mat, d_minus))
    return(normalised_lap_mat)

def smallest_eigenvalue(m1):
    return(abs(min(calculate_eigenvalues(m1))))

def average_eigenvalue(m1):
    return(np.mean(calculate_eigenvalues(m1)))

def embed(m1):
    vec = [calculate_average_distances(m1),
     calculate_biggest_eigengap(m1), abs(algebraic_connectivity(m1))]
    return(vec)
