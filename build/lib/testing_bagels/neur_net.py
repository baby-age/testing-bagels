import numpy as np
from numpy import dot

import testing_bagels.embedder as emb
import testing_bagels.graphgen as gen
from sklearn import preprocessing

eps = 0.001
regularity = 0.001

def d_tanh(x):
    return(1-np.power(np.tanh(x),2))

class NN():
    def __init__(self, signature):
        self.dimension_signature = signature
        self.num_layers = len(signature)

        self.weights = self.initialise_weights()
        self.biases = self.initialise_biases()

    def initialise_weights(self):
        weights = []
        for i in range(1, self.num_layers):
            dim_minus = self.dimension_signature[i-1]
            dim = self.dimension_signature[i]

            weights.append(np.random.randn(dim_minus,
                dim) / np.sqrt(dim_minus))
        return(weights)

    def initialise_biases(self):
        biases = []
        for i in range(1, self.num_layers):
            dim = self.dimension_signature[i]
            bias = np.zeros((1, dim))
            biases.append(bias)

        return(biases)

    def feedforward(self, vec):
        self.outputs = [vec]
        for w, b in zip(self.weights, self.biases):
            vec = np.tanh(np.dot(vec, w) + b)
            self.outputs.append(vec)
        return(vec)

    def predict(self, vec):
        self.outputs = []
        for w, b in zip(self.weights[0:-1], self.biases[0:-1]):
            vec = np.tanh(np.dot(vec, w) + b)
            self.outputs.append(vec)

        vec = np.dot(vec, self.weights[-1]) + self.biases[-1]
        self.outputs.append(vec)
        return(vec)

    def calculate_simple_error(y, predicted):
        return 0.5*(predicted-y)

    def calculate_error_with_exponent(self, y, predicted, exponent):
        return 0.5*np.sign(predicted-y)*np.power(predicted-y, exponent)

    def backpropagate(self, X, y, predicted):

        self.error = self.calculate_error_with_exponent(y, predicted, 2)
        self.delta = self.error * d_tanh(self.outputs[self.num_layers-1])

        deltas = {}
        deltas[self.num_layers-1] = self.delta

        delta = self.delta
        for i in range(self.num_layers-2, 0, -1):
            hid_error = np.dot(delta, np.transpose(self.weights[i]))
            delta = hid_error * d_tanh(self.outputs[i])
            deltas[i] = delta

        dbs = {}
        for i in range(1, self.num_layers):
            dbs[i] = np.sum(deltas[i], axis = 0)

        dws = {}
        for i in range(1, self.num_layers):
            dws[i] = np.dot(np.transpose(self.outputs[i-1]), deltas[i])
            dws[i] += regularity * self.weights[i-1]

        for i in range(self.num_layers-1):
            self.weights[i] += -eps * dws[i+1]
            self.biases[i] += -eps * dbs[i+1]


    def train_network(self, X, y):
        predicted = self.feedforward(X)
        self.backpropagate(X, y, predicted)


data = [[0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1],
        [0.4, 0.5, 0.5],
        [0.5, 0.6, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [1.0, 0.5, 0.5],
        [1.0, 1.0, 0.5],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]]

labels = [[0.2], [0.21], [0.2], [0.35], [0.41], [0.4], [0.38], [0.5], [0.6], [0.8], [0.82], [0.8], [0.8]]

train_data = gen.generate_graphs(400, 20, 2)
test_data = gen.generate_graphs(200, 20, 2)

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

#model = train_model(new_train_data, new_train_labels, 2)

neur = NN([3, 5, 5, 1])

mean = 11
divisor = 8
normalized_labels = np.divide(np.subtract(new_train_labels, mean), divisor)

num_iterations = 1000

for i in range(num_iterations):
    lower_bound = np.random.randint(0, int(0.3*num_iterations))
    upper_bound = np.random.randint(int(0.6*num_iterations), num_iterations)
    neur.train_network(new_train_data[lower_bound:upper_bound], normalized_labels[lower_bound:upper_bound])

tr = np.add(divisor*neur.predict(new_test_data), mean)
print("Predicted:\n", np.add(divisor*neur.predict(new_test_data), mean)[1:10],
      "\nActual:\n", np.matrix(new_test_labels)[1:10],
      "\nDiff:\n", np.subtract(tr, new_test_labels)[1:10],
      "\nStandard deviation: ", np.std(a = np.subtract(tr, new_test_labels)),
      "\nMean diff:\t    ", np.mean(a = np.subtract(tr, new_test_labels)),
      "\nMean diff abs:\t    ", np.mean(a = abs(np.subtract(tr, new_test_labels))))
