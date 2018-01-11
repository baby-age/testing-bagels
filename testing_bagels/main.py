import pandas as pand
import numpy as np
import testing_bagels.embedder as emb
import testing_bagels.graphgen as gen
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model

def visualise(m, y):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(m[:,0],m[:,1],m[:,2], c = train_data['y'])
    ax.set_xlabel('Ave. dist.')
    ax.set_ylabel('Max. eig.gap')
    ax.set_zlabel('Ave. eig')

    plt.show()

train_data = gen.generate_graphs(300, 60, 100)
test_data = gen.generate_graphs(100, 60, 100)

new_train_data = []
new_test_data = []

for i in train_data['X']:
    new_train_data.append(emb.embed(i))

for i in test_data['X']:
    new_test_data.append(emb.embed(i))

clf = linear_model.Lasso(alpha=0.1)

clf.fit(new_train_data, train_data['y'])

pred = clf.predict(new_test_data)

print("Average error: ", sum(pred-test_data['y'])/len(test_data))

visualise(np.matrix(new_train_data), train_data['y'])
