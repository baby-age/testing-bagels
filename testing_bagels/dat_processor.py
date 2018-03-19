import pandas as pand
import numpy as np
import math
from numpy import array_split
import testing_bagels.matrix_preprocessor as mp

# Algorithm takes just the upper part of the triangle matrix
def flatten_data(data):
    new_train_X = []
    for i in data['X']:
        flattened_matrix = []
        mp.preprocess_matrix(i)
        for k in range(len(i)):
            row = i[k][1+k:len(i)]
            flattened_matrix = np.concatenate([flattened_matrix,row])
            #new_train_X.append(i.flatten())
        new_train_X.append(flattened_matrix)
    new_train_Y = []
    for i in data['y']:
        new_train_Y.append(i)

    d = {'X': new_train_X, 'y':new_train_Y}
    df = pand.DataFrame(d)
    return(df)

def listify(data):
    return list(data['X']), list(data['y'])

def to_train_test(X, y, train_n):
    train_indices = np.random.choice(len(X), train_n, replace = False)
    train_data, train_label, test_data, test_label = [], [], [], []

    for i in range(0,len(X)):
        if i in train_indices:
            train_data.append(X[i])
            train_label.append(y[i])
        else:
            test_data.append(X[i])
            test_label.append(y[i])

    return train_data, train_label, test_data, test_label

def from_regression_to_classify(y, tresholds):
    for index, item in enumerate(y):
        for i, tresh in enumerate(tresholds):
            if item >= tresh[0] and item < tresh[1]:
                y[index] = i
                break
    return y

def to_absolute(y):
    for index, item in enumerate(y):
        y[index] = abs(item)

    return y

def reduce_graph_regions(X, new_dim):
    reduced_graphs = []
    for i in X:
        reduced_graphs.append(reduce_size(i, new_dim))
    return np.asarray(reduced_graphs)

def reduce_size(graph, new_dim):

    return 0
