import numpy as np
import os
from pylab import *
from sklearn import linear_model
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.svm import SVR
import testing_bagels.embedder as emb
import testing_bagels.graphgen as gen
import testing_bagels.graphread as read
import testing_bagels.neur_net as neural
import testing_bagels.visualiser as vis
import testing_bagels.dat_processor as dat_proc
import testing_bagels.pca as pca
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


wd = os.getcwd() + "/Data/"

# Put correct path!
csv_path = os.getcwd() + "/Neuro_at_fullterm_age.csv"

data = read.read_data(path = wd, group = 'preterm', modality = 'PPC',
    frequency_range = 'theta', csv_path = csv_path)

X, y = dat_proc.listify(dat_proc.flatten_data(data))

X_embedded = TSNE(n_components=3, perplexity=40.0,
    method='exact', n_iter = 4000, init='random', n_iter_without_progress = 1000, learning_rate = 30, early_exaggeration=50, metric='hamming').fit_transform(X)

train_data, train_label, test_data, test_label = dat_proc.to_train_test(X,
    y, 20)


col = []
for y in to_y:
    c = math.floor(5*(y+1))
    col.append(c)

reg.fit(train_data, train_label)

predicted = list(reg.predict(test_data))
array_predicted = np.asarray(predicted)
array_test_labels = np.asarray(test_label)

print("Predicted:", array_predicted, "\n\nReal:", array_test_labels,
    "\n\n\nDiff:", array_predicted-array_test_labels)
