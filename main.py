import numpy as np
import os
from pylab import *
from sklearn import linear_model
from sklearn import manifold
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.svm import LinearSVR
import testing_bagels.embedder as emb
import testing_bagels.feature_calculator as fc
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor
import sklearn.metrics as metrics
#import testing_bagels.autoencoder as auto
import testing_bagels.graphgen as gen
import testing_bagels.graphread as read
import testing_bagels.neur_net as neural
import testing_bagels.visualiser as vis
import testing_bagels.dat_processor as dat_proc
import testing_bagels.matrix_preprocessor as mp
import testing_bagels.pca as pca
import testing_bagels.predictor as predictor
import testing_bagels.average_brain_area_connections as ave
import warnings
import tensorflow as tf
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV, cross_val_score
from tpot import TPOTRegressor
from tpot import TPOTClassifier
from sklearn import linear_model


warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

wd = os.getcwd() + "/Data/"

# Put correct path!
csv_path = os.getcwd() + "/Neuro_at_fullterm_age.csv"

prep_data1 = read.read_data(path = wd, group = 'preterm', modality = 'PPC',
    frequency_range = 'theta', csv_path = csv_path)
prep_data2 = read.read_data(path = wd, group = 'preterm', modality = 'PPC',
        frequency_range = 'beta', csv_path = csv_path)
prep_data3 = read.read_data(path = wd, group = 'preterm', modality = 'PPC',
                frequency_range = 'alpha', csv_path = csv_path)
prep_data4 = read.read_data(path = wd, group = 'preterm', modality = 'PPC',
                frequency_range = 'delta', csv_path = csv_path)

Xs1, y = dat_proc.listify(dat_proc.flatten_data(prep_data1))
Xs2, y = dat_proc.listify(dat_proc.flatten_data(prep_data2))
Xs3, y = dat_proc.listify(dat_proc.flatten_data(prep_data3))
Xs4, y = dat_proc.listify(dat_proc.flatten_data(prep_data4))

X1, y = dat_proc.listify(dat_proc.flatten_data(prep_data1))
X2, y = dat_proc.listify(dat_proc.flatten_data(prep_data2))
X3, y = dat_proc.listify(dat_proc.flatten_data(prep_data3))
X4, y = dat_proc.listify(dat_proc.flatten_data(prep_data3))


X1 = np.array([[ emb.algebraic_connectivity(emb.from_triangular_to_matrix(x))] for x in X1])


X2 = np.array([[ emb.algebraic_connectivity(emb.from_triangular_to_matrix(x))] for x in X2])

X3 = np.array([[emb.algebraic_connectivity(emb.from_triangular_to_matrix(x))] for x in X3])

X4 = np.array([[emb.algebraic_connectivity(emb.from_triangular_to_matrix(x))] for x in X4])


X = np.column_stack((X1, X2, X3, X4))

Xs = np.column_stack((Xs1, Xs2, Xs3))
#X = np.column_stack((X1, X2, X3, X4))

#y = dat_proc.to_absolute(y)

#ENABLE THESE TO USE CLASSIFICATION
#y = dat_proc.from_regression_to_classify(y, [[-10,-0.0], [-0.0, 10.0]])
#le = preprocessing.LabelEncoder()
#le.fit(y)
#y = le.transform(y)

X_df = prep_data1['X']
y_df = prep_data1['y']

#vis.visualize_3d(X, y)

res = fc.modularity_and_efficiency(X_df)

modes1 = [list(x) for x in zip(res[0], res[1])]

X_embedded = manifold.SpectralEmbedding(n_components=3).fit_transform(Xs1)
X_embedded = np.column_stack((X_embedded, X1, X2, X3, X4))

#vis.visualize_3d(X_embedded, y)

tpoptimizer = make_pipeline(
    RBFSampler(gamma=0.25, n_components=100),
    GradientBoostingRegressor(learning_rate=0.01, max_depth=4, min_samples_split=2, subsample=0.5, n_estimators=1000, verbose=0)
)
clf1 = DummyRegressor(strategy='mean')
clf2 = DummyRegressor(strategy='median')

total_our = 0
total_dummy1 = 0
total_dummy2 = 0
times = 0
for value in range(0, 100):
    train_data, train_label, test_data, test_label = dat_proc.to_train_test(X_embedded,
        y, 28)

    tpoptimizer.fit(np.array(train_data), np.array(train_label))
    clf1.fit(np.array(train_data), np.array(train_label))
    clf2.fit(np.array(train_data), np.array(train_label))

    total_our = total_our + metrics.mean_squared_error(tpoptimizer.predict(np.array(test_data)), test_label)


    total_dummy1 = total_dummy1 + metrics.mean_squared_error(clf1.predict(np.array(test_data)), test_label)
    total_dummy2 = total_dummy2 + metrics.mean_squared_error(clf2.predict(np.array(test_data)), test_label)
    times = times+1

print("Our:", total_our/times)
print("Dummy mean: ", total_dummy1/times)
print("Dummy median: ", total_dummy2/times)


#tpoptimizer = TPOTClassifier(generations = 200, population_size = 20, verbosity =2, cv=4, config_dict=None)




#tpoptimizer.fit(np.array(train_data), np.array(train_label))

#print(tpoptimizer.score(np.array(test_data), np.array(test_label)))
print("Predicted: ", tpoptimizer.predict(np.array(test_data)))
print("Real:", test_label)
#tpoptimizer.export('tpot_exported_pipeline.py')

#exported_pipeline = make_pipeline(
#    RobustScaler(),
#    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
#    LinearSVR(C=1e-05, epsilon=0.0001, tol=0.1)
#)
#fit = exported_pipeline.fit(train_data, train_label)
#results = fit.predict(test_data)
#print("Score:", fit.score(np.array(test_data), np.array(test_label)))
#print(results)
#print(test_label)
