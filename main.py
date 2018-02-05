import pandas as pand
import numpy as np
import testing_bagels.embedder as emb
import testing_bagels.graphgen as gen
import testing_bagels.graphread as read
import matplotlib.pyplot as plt
import testing_bagels.visualiser as vis
from matplotlib import pylab
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.decomposition import PCA
import os
import testing_bagels.neur_net as neural
from sklearn.kernel_ridge import KernelRidge


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

clf = KernelRidge(alpha=1.0)
clf.fit(to_X[1:10], to_y[1:10])
print(clf.get_params)
predicted = clf.predict(to_X[11:20])
print(predicted)

components = pca.components_
print(sum(explained_variances[0:10]))


vis.plot_PCs(components)
