import numpy as np
from numpy import dot

import testing_bagels.embedder as emb
import testing_bagels.graphgen as gen

eps = 0.1
regularity = 0.1

def tanh(x):
    return((np.exp(x) - np.exp(-x))/(np.exp(-x) + np.exp(x)))

def d_tanh(x):
    return(4/np.power(np.exp(-x) + np.exp(x), 2))

class NN():
    def __init__(self, signature):
        self.dimension_signature = signature
        self.num_layers = len(signature)

        self.weights = self.initialise_weights()
        self.biases = self.initialise_biases()

        print(self.biases)

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
            biases.append(np.zeros((1, dim)))
        return(biases)


    def backpropagate(self, X, labels, predicted):

        diff = predicted - labels

        self.error = 0.5 * np.power(diff, 2)
        self.delta = self.error * d_tanh(predicted)

        db2 = np.sum(self.delta, axis=0, keepdims=True)


        hid_error = np.dot(self.delta, np.transpose(self.weights[1]))
        hid_delta = hid_error*d_tanh(self.outputs[0])

        db1 = np.sum(hid_delta, axis=0)

        dW1 = np.dot(np.transpose(X), hid_delta)
        dW2 = np.dot(np.transpose(self.outputs[0]), self.delta)

        dW1 += regularity * self.weights[0]
        dW2 += regularity * self.weights[1]

        self.weights[0] += -eps * dW1
        self.biases[0] += -eps * db1
        self.weights[1] += -eps * dW2
        self.biases[1] += -eps * db2





    def feedforward(self, vec):
        self.outputs = []
        for w, b in zip(self.weights, self.biases):
            vec = np.tanh(np.dot(vec, w) + b)
            self.outputs.append(vec)
        return(vec)

    def train_network(self, X, y):
        predicted = self.feedforward(X)
        self.backpropagate(X, y, predicted)


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

#model = train_model(new_train_data, new_train_labels, 2)

neur = NN([3, 3, 1])

for i in range(2):
    neur.train_network(new_train_data, new_train_labels)
