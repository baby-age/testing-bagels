from matplotlib import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pand
from pylab import *
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.manifold import TSNE
import testing_bagels.embedder as emb
import testing_bagels.graphgen as gen
import testing_bagels.graphread as read
import testing_bagels.neur_net as neural
import testing_bagels.visualiser as vis


def flatten_data(data):
    new_train_X = []
    for i in train_data['X']:
        new_train_X.append(i.flatten())
    new_train_Y = []
    for i in train_data['y']:
        new_train_Y.append(i)

    d = {'X': new_train_X, 'y':new_train_Y}
    df = pand.DataFrame(d)
    return(df)

wd = os.getcwd() + "/Data/"

# Put correct path!
csv_path = os.getcwd() + "/Neuro_at_fullterm_age.csv"

train_data = read.read_data(path = wd, group = 'preterm', modality = 'PPC',
frequency_range = 'theta', csv_path = csv_path)

new_train_data = flatten_data(train_data)

to_X = list(new_train_data['X'])
to_y = list(new_train_data['y'])

pca = PCA(n_components=400)
pca.fit(to_X)
explained_variances = pca.explained_variance_ratio_

components = pca.components_
print("The first 20 components explain ", sum(explained_variances[0:19]),
    " percent of the variance.")

X_embedded = TSNE(n_components=2).fit_transform(to_X)
print(X_embedded)
print(X_embedded[:,0])
plt.scatter(X_embedded[:,0], X_embedded[:,1])
plt.show()
#vis.plot_PCs(components)
