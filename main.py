import pandas as pand
import numpy as np
import testing_bagels.embedder as emb
import testing_bagels.graphgen as gen
import testing_bagels.graphread as read
import matplotlib.pyplot as plt
from matplotlib import pylab
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.decomposition import PCA
import os

wd = os.getcwd() + "/Data/"

train_data = read.read_data(path = wd, group = 'preterm', modality = 'PPC', frequency_range = 'theta')


new_train_data = []
for i in train_data['X']:
    new_train_data.append(i.flatten())

pca = PCA(n_components=400)
pca.fit(new_train_data)
explained_variances = pca.explained_variance_ratio_

components = pca.components_
print(sum(explained_variances[0:10]))
fst = components[0]
fst = fst.reshape((58, 58))
imshow(fst, interpolation='nearest')
plt.show()
