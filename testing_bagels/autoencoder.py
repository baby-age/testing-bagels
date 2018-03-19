from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

"""
def save_model(model, name = 'bagel-autoencoder'):
    model_name = name
    model.summary()
    model.save_weights('%s.h5' % model_name, overwrite=True)
    model_json = model.to_json()
    with open('%.json' % model_name, "w") as json_file
"""

"""
Input for example 90 % of graphs for training and 10 % of graphs for testing.
Outputs tuple with encoded graphs as a list of 32-arrays and decoder which
can decode arrays back to 58x58 graphs (in array form).
Use decoder.predict(encoded_graphs).
"""
def process_data(data, data_test):
    encoding_dim = 16
    graph_dim = len(data[-1])

    input_graph = Input(shape=(graph_dim,))

    encoded = Dense(1024, activation='relu')(input_graph)
    encoded = Dense(512, activation='relu')(encoded)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(1024, activation='relu')(decoded)
    decoded = Dense(graph_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_graph, decoded)
    encoder = Model(input_graph, encoded)
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adagrad', loss='binary_crossentropy')

    x_train = np.asarray(data)
    x_test = np.asarray(data_test)

    autoencoder.fit(x_train, x_train,
                    epochs=300,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    encoded_graphs = encoder.predict(x_test)

    return encoded_graphs, decoder
