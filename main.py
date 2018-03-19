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
from sklearn import preprocessing
from sklearn.svm import LinearSVR
import testing_bagels.embedder as emb
import testing_bagels.feature_calculator as fc
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

Xs, y = dat_proc.listify(dat_proc.flatten_data(prep_data1))

X1, y = dat_proc.listify(dat_proc.flatten_data(prep_data1))
X2, y = dat_proc.listify(dat_proc.flatten_data(prep_data2))
X3, y = dat_proc.listify(dat_proc.flatten_data(prep_data3))
X4, y = dat_proc.listify(dat_proc.flatten_data(prep_data3))


X1 = np.array([[emb.smallest_eigenvalue(emb.from_triangular_to_matrix(x)),emb.biggest_eigenvalue(emb.from_triangular_to_matrix(x)), emb.algebraic_connectivity(emb.from_triangular_to_matrix(x))] for x in X1])


X2 = np.array([[emb.smallest_eigenvalue(emb.from_triangular_to_matrix(x)), emb.biggest_eigenvalue(emb.from_triangular_to_matrix(x)),emb.algebraic_connectivity(emb.from_triangular_to_matrix(x))] for x in X2])

X3 = np.array([[emb.smallest_eigenvalue(emb.from_triangular_to_matrix(x)),emb.biggest_eigenvalue(emb.from_triangular_to_matrix(x)), emb.algebraic_connectivity(emb.from_triangular_to_matrix(x))] for x in X3])

X4 = np.array([[emb.smallest_eigenvalue(emb.from_triangular_to_matrix(x)),emb.biggest_eigenvalue(emb.from_triangular_to_matrix(x)), emb.algebraic_connectivity(emb.from_triangular_to_matrix(x))] for x in X4])


X = np.column_stack((X1, X2, X3, X4))
#X = np.column_stack((X1, X2, X3, X4))

#y = dat_proc.to_absolute(y)

#ENABLE THESE TO USE CLASSIFICATION
#y = dat_proc.from_regression_to_classify(y, [[-10,0], [0, 10.0]])
#le = preprocessing.LabelEncoder()
#le.fit(y)
#y = le.transform(y)

X_df = prep_data1['X']
y_df = prep_data1['y']
#vis.visualize_3d(X, y)

res = fc.modularity_and_efficiency(X_df)

modes1 = [list(x) for x in zip(res[0], res[1])]


train_data, train_label, test_data, test_label = dat_proc.to_train_test(X,
    y, 28)

tpoptimizer = TPOTRegressor(generations = 100, population_size = 20, verbosity =2, cv=4, config_dict="TPOT light")

tpoptimizer.fit(np.array(train_data), np.array(train_label))

print(tpoptimizer.score(np.array(test_data), np.array(test_label)))
print("Predicted: ", tpoptimizer.predict(np.array(test_data)))
print("Real:", test_label)
tpoptimizer.export('tpot_exported_pipeline.py')

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
