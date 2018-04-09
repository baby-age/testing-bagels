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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
import testing_bagels.embedder as emb
import testing_bagels.feature_calculator as fc
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_selection import VarianceThreshold
import sklearn.metrics as metrics
import testing_bagels.autoencoder as auto
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


X1 = np.array([[emb.algebraic_connectivity(emb.from_triangular_to_matrix(x))] for x in X1])


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

#y = np.abs(y)
#vis.visualize_3d(X, y)

#res = fc.modularity_and_efficiency(X_df)

#modes1 = [list(x) for x in zip(res[0], res[1])]

#X_embedded = manifold.SpectralEmbedding(n_components=4).fit_transform(Xs1)

#X_embedded = np.column_stack((X_embedded, X1))

#vis.visualize_3d(X_embedded, y)

"""
tpoptimizer = make_pipeline(
    RBFSampler(gamma=0.25),
    GradientBoostingRegressor(learning_rate=0.1, max_depth=4, min_samples_split=2, subsample=0.8, n_estimators=30, loss="huber", verbose=0, alpha=0.3)
)
"""
"""
tpoptimizer = make_pipeline(
    RBFSampler(gamma=0.25),
    RandomForestRegressor(max_depth=10, n_estimators=80, verbose=0, criterion="mae")
)
"""


"""
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

"""

def column(matrix, i):
    return [row[i] for row in matrix]

#Xs1, components = pca.pca_transform(Xs1)

sel = SelectKBest(f_regression, k=4)




#train_data = sel.fit_transform(train_data, train_label)
#print(sel.get_support(indices=True))

#idxs_selected = sel.get_support(indices=True)
#idxs_selected = range(50,200)
idxs_selected = [616, 120, 855, 856, 1057,1143, 168, 156]
Xs1 = column(Xs1, idxs_selected)
train_data, train_label, test_data, test_label = dat_proc.to_train_test(Xs1,
    y, 28)

#idxs_selected = [9, 120, 156, 168, 171, 616, 855, 856, 1057, 1143]
#test_data = sel.transform(test_data)



scaler = preprocessing.StandardScaler().fit(Xs1)
train_data = scaler.fit_transform(train_data)
test_data = scaler.fit_transform(test_data)
Xs1 = scaler.fit_transform(Xs1)

Xs1, components = pca.pca_transform(Xs1)
vis.visualize_2d(Xs1, y)


tpoptimizer = linear_model.BayesianRidge()

#tpoptimizer = KernelRidge(alpha=1, degree=3).fit(train_data, train_label)

#tpoptimizer = TPOTRegressor(generations = 100, population_size = 10, verbosity =2, cv=4, config_dict="TPOT light")




tpoptimizer.fit(np.array(train_data), np.array(train_label))
#print(tpoptimizer.score(np.array(test_data), np.array(test_label)))
print("Predicted: ", tpoptimizer.predict(np.array(test_data)))
print("Real:", test_label)
print("MSE:", metrics.mean_squared_error(y_pred=tpoptimizer.predict(np.array(test_data)), y_true=test_label))
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
