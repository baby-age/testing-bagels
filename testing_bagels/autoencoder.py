from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution1D
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import Activation
from keras import regularizers
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
    encoding_dim = 8
    graph_dim = len(data[-1])

    input_graph = Input(shape=(graph_dim,))

    encoded = Dense(512, activation='relu')(input_graph)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = BatchNormalization()(encoded)
    decoded = Activation('tanh')(decoded)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(graph_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_graph, decoded)
    encoder = Model(input_graph, encoded)
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='mse')

    x_train = np.asarray(data)
    x_test = np.asarray(data_test)

    autoencoder.fit(x_train, x_train,
                    epochs=20,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    encoded_train = encoder.predict(x_train)
    encoded_graphs = encoder.predict(x_test)

    return encoded_train, encoded_graphs, decoder
