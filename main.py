import numpy as np
import os
from pylab import *
from sklearn import linear_model
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neural_network import MLPRegressor
import testing_bagels.embedder as emb
import testing_bagels.graphgen as gen
import testing_bagels.graphread as read
import testing_bagels.neur_net as neural
import testing_bagels.visualiser as vis
import testing_bagels.dat_processor as dat_proc
import testing_bagels.pca as pca
import testing_bagels.predictor as predictor
import warnings
import tensorflow as tf
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV, cross_val_score

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")



wd = os.getcwd() + "/Data/"

# Put correct path!
csv_path = os.getcwd() + "/Neuro_at_fullterm_age.csv"

data = read.read_data(path = wd, group = 'preterm', modality = 'PPC',
    frequency_range = 'theta', csv_path = csv_path)

X, y = dat_proc.listify(dat_proc.flatten_data(data))

X_embedded = TSNE(n_components=2, perplexity=2000000.0, n_iter = 40000,
    init='random', n_iter_without_progress = 1000).fit_transform(X)

#y = dat_proc.from_regression_to_classify(y)

#vis.visualize_3d(X_embedded, y)
train_data, train_label, test_data, test_label = dat_proc.to_train_test(X_embedded,
    y, 26)

svr = MLPRegressor(max_iter = 1000000)
Cs = np.logspace(-12, -9, 40)
clf = GridSearchCV(estimator=svr, param_grid=dict(alpha=Cs), n_jobs=-1)


#model = predictor.build_model(train_data, train_label)
model = clf.fit(train_data, train_label)

print(clf.best_score_)
array_predicted = model.predict(test_data)
array_real = test_label

score = model.score(test_data, test_label)

print("\n\nPredicted:", array_predicted)
print("\n\nReal:", array_real)
print("\n\nScore:", score)
