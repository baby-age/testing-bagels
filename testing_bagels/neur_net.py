import numpy as np
from numpy import dot

import testing_bagels.embedder as emb
import testing_bagels.graphgen as gen

dimension_signature = [3, 3, 1]
num_layers = len(dimension_signature)

eps = 0.1
regularity = 0.1

def backpropagate(weights, biases, error):

    return(weights, biases)



def train_model(data, labels, num_iterations):
    #Initialising the model with empty weights and bias vectors.
    model = {'weights':[], 'bias':[]}

    weights = model['weights']
    biases = model['bias']

    #Random initial values for weight matrices.
    for i in range(1, num_layers):
        weights.append(np.random.randn(dimension_signature[i-1],
          dimension_signature[i]) / np.sqrt(dimension_signature[i-1]))
        biases.append(np.zeros((1, dimension_signature[i])))


    for i in range(num_iterations):
        z = {'z_in':[], 'z_out':[]}

        z['z_out'].append(data)

        for i in range(num_layers-1):
            to_append = np.dot(z['z_out'][i], weights[i])
            to_append = to_append + biases[i]
            z['z_in'].append(to_append)
            z['z_out'].append(np.tanh(to_append))

        fin = z['z_in'][num_layers-2]
        scores = np.exp(fin)
        diff = scores-labels
        error = 0.5 * np.power(diff, 2)

        print(error)

        #TODO Backpropagation part.
        weights, biases = backpropagate(weights, biases, error)

        model = {'weights': weights, 'bias': biases}

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
