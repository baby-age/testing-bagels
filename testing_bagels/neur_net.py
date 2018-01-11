import numpy as np
from numpy import dot

import testing_bagels.embedder as emb
import testing_bagels.graphgen as gen

dimension_signature = [3, 3, 1]
num_layers = len(dimension_signature)

e = 0.1
regularity = 0.1

def backpropagate(weights, affines, error):
    return(weights, affines)

def train_model(data, labels, num_iterations):
    model = {'weights':[], 'affines':[]}

    weights = model['weights']
    affines = model['affines']

    for i in range(1, num_layers):
        weights.append(np.random.randn(dimension_signature[i-1],
          dimension_signature[i]) / np.sqrt(dimension_signature[i-1]))
        affines.append(np.zeros((1, dimension_signature[i])))


    for i in range(num_iterations):
        z_in = []
        z_out = []
        z_out.append(data)

        for i in range(num_layers-1):
            to_append = np.dot(z_out[i], weights[i])
            to_append = to_append + affines[i]
            z_in.append(to_append)
            z_out.append(np.tanh(to_append))

        fin = z_in[num_layers-2]
        scores = np.exp(fin)
        scores = scores[:,0]

        labels = np.asarray([item for sublist in labels for item in sublist])

        error = 0.5 * np.power(scores - labels, 2)

        print(error)

        #TODO Backpropagation part.
        weights, affines = backpropagate(weights, affines, error)

        model = {'weights': weights, 'affines': affines}

    return model


train_data = gen.generate_graphs(200, 4, 90)
test_data = gen.generate_graphs(100, 4, 90)

new_train_data = []
new_train_labels = []
new_test_data = []
new_test_labels = []

for i in train_data['X']:
    new_train_data.append(emb.embed(i))

for i in train_data['y']:
    new_train_labels.append([i])

for i in test_data['X']:
    new_test_data.append(emb.embed(i))

for i in test_data['y']:
    new_test_labels.append([i])

model = train_model(new_train_data, new_train_labels, 1)
