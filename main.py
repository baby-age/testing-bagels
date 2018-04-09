import numpy as np
import os
from pylab import *
from sklearn import linear_model
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn import preprocessing
import testing_bagels.embedder as emb
import testing_bagels.graphgen as gen
import testing_bagels.graphread as read
import testing_bagels.neur_net as neural
import testing_bagels.visualiser as vis
import testing_bagels.dat_processor as dat_proc
import testing_bagels.pca as pca
import testing_bagels.predictor as predictor
import testing_bagels.matrix_preprocessor as mp
import testing_bagels.feature_calculator as feat
from testing_bagels.average_brain_area_connections import get_mean_strength_of_connections
import warnings
import tensorflow as tf
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from tpot import TPOTRegressor
import scipy.io
from scipy.stats.stats import pearsonr
from sklearn.dummy import DummyRegressor as dum
from sklearn.dummy import DummyClassifier as dumcla

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MaxAbsScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.feature_selection import SelectPercentile, f_regression


warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

wd = os.getcwd() + "/Data/"

# Put correct path!
csv_preterm = "/home/local/tuletule/Documents/opiskelu/vauvaika/Helsinki_SVM/Neuro_at_fullterm_age.csv"
data = read.read_data(path = wd, group = 'preterm', modality = 'PPC',
    frequency_range = 'theta', csv_path = csv_preterm, sleep_mode = 'active sleep')

csv_control = "/home/local/tuletule/Documents/opiskelu/vauvaika/Neuro_Controls.mat"
control_data = read.read_data(path = wd, group = 'control', modality = 'PPC',   
    frequency_range = 'theta', csv_path = csv_control, sleep_mode = 'active sleep')

# print(data)
# print(control_data)

# prepr_data = pd.DataFrame(data, columns = ['X'])
# prepr_data['y'] = data['y']

# data = data.append(control_data, ignore_index = True)
# print(all_data)
# print(data)
prepr_data1 = mp.preprocess_whole_data(data)
prepr_data2 = mp.preprocess_whole_data(control_data)

# X, y = dat_proc.listify(dat_proc.flatten_data(data))
X1, y1 = dat_proc.listify(dat_proc.flatten_data(prepr_data1))

neopath = '/home/local/tuletule/Documents/opiskelu/vauvaika/neoNets/consistent_network.m'
neo = scipy.io.loadmat(neopath)

print(neo)

# z1 =  np.repeat(0, prepr_data1.shape[0])

# X2, y2 = dat_proc.listify(dat_proc.flatten_data(prepr_data2))
# z2 = np.repeat(1, prepr_data2.shape[0]) 
# y = np.concatenate((z1, z2))
# X = np.row_stack((X1, X2))

# le = preprocessing.LabelEncoder()
# le.fit(y)
# y = le.transform(y)

X_df = prepr_data1['X']
y_df = prepr_data1['y']

y_df = y_df.astype('float')

path_to_atlas = "/home/local/tuletule/Documents/opiskelu/vauvaika/Helsinki_SVM/MyAtlas_n58.mat"
connections = get_mean_strength_of_connections(prepr_data1, path_to_atlas)
f_connections = connections['Frontal_connections'].as_matrix()

connections = connections.as_matrix()

print(pearsonr(f_connections, y1))


# for x in X:
#   x = np.asmatrix(x).reshape((58,58))
#   print(x, emb.algebraic_connectivity(x))
# connectivity_matrices = [emb.algebraic_connectivity(x) for x in X]
# print(connectivity_matrices)

# variances, comps = pca.build_pca(X)

mod_eff = feat.modularity_and_efficiency(X_df)
mod = mod_eff[0]
eff = mod_eff[1]
print(pearsonr(mod, y1))
print(pearsonr(eff, y1))


# predicting_scores = [list(x) for x in zip(mod_eff[0], mod_eff[1])]

# clusters = feat.local_clustering(X_df)

# train_data, train_label, test_data, test_label = dat_proc.to_train_test(X, y, 27)
# train_data, train_label, test_data, test_label = dat_proc.to_train_test(predicting_scores, y, 28)
# train_data, train_label, test_data, test_label = dat_proc.to_train_test(mod_eff[0], y, 27)

# semb = Isomap(n_components = 3)
# embedded = semb.fit_transform(X)

"""
plt.scatter(embedded[:,0], embedded[:,1], c = y)
plt.show()
"""
# vis.visualize_3d(embedded, y)

# exported_pipeline = make_pipeline(
#     RBFSampler(gamma=0.25, n_components=80),
#     # SelectPercentile(score_func=f_regression, percentile=29),
#     RandomForestClassifier(n_estimators=80, max_depth=10)
# )

# processed_data = embedded
# processed_data = X

# dum_avg = 0
# res_avg = 0
# n = 20

# for i in range(n):
#     train_data, train_label, test_data, test_label = dat_proc.to_train_test(processed_data, y, int(len(processed_data)*0.8))
#     results = predictor.predict_and_score(train_data, train_label, test_data, test_label, exported_pipeline)
#     dum_preds = predictor.predict_and_score(train_data, train_label, test_data, test_label, dumcla())

#     print(test_label)
#     print(results[0])

#     dum_avg = dum_avg + dum_preds[1]
#     res_avg = res_avg + results[1]

# dum_avg = dum_avg / n
# res_avg = res_avg / n

# print(dum_avg)
# print(res_avg)

# processed_data = connections
# processed_data = clusters
# processed_data
# processed_data = embedded
# train_data, train_label, test_data, test_label = dat_proc.to_train_test(processed_data, y, int(len(processed_data)*0.8))

# tpoptimizer = TPOTRegressor(generations = 20, population_size = 30, verbosity = 2)
# tpoptimizer.fit(np.array(train_data), np.array(train_label))
# results = tpoptimizer.predict(np.array(test_data))
# # print(tpoptimizer.score(np.array(test_data), np.array(test_label)))
# tpoptimizer.export('tpot_exported_pipeline.py')

# predictor.tune_SVR_params(train_data, train_label, test_data, test_label)

# exported_pipeline.fit(train_data, train_label)
# results = exported_pipeline.predict(test_data)
# print(metrics.mean_squared_error(test_label, results))

# lsvr = svm.LinearSVR(C=0.01, dual=True, epsilon=0.01, loss="epsilon_insensitive", tol=0.01)
# svr = svm.SVR(C = 100, kernel = "linear", epsilon = 0.01)
# svr = svm.SVR(C = 50000, kernel = "linear", epsilon = 0.05)
# svr = svm.SVR(gamma = 0.001, C = 1, kernel = "rbf", epsilon = 3)
# svr_preds = svr.fit(train_data, train_label).predict(test_data)
# print(metrics.mean_squared_error(test_label, svr_preds))
# print()
# for idx, label in enumerate(test_label):
#     print(label, svr_preds[idx])
#     # print(label, results[idx], svr_preds[idx])

# print()
# print(feat.modularity_and_efficiency(data['X']))

# dum_preds = predictor.predict_and_score(train_data, train_label, test_data, test_label, dum())
# svr_preds = predictor.predict_and_score(train_data, train_label, test_data, test_label, svr)
# lr_preds = predictor.predict_and_score(train_data, train_label, test_data, test_label)
# lsvr_preds = predictor.predict_and_score(train_data, train_label, test_data, test_label, lsvr)
# print(test_label)
# print()
# print(dum_preds[0])
# print(results)
# print(svr_preds[0])
# print(lr_preds[0])
# print(lsvr_preds[0])

# print(dum_preds[1])
# print(metrics.mean_squared_error(test_label, results))
# print(lsvr_preds[1])
# print(svr_preds[1])
# print(lr_preds[1])

#predictor.tune_SVR_params(train_data, train_label, test_data, test_label)

'''
train_data_comb = []
test_data_comb = []
train_data_mod = []
test_data_mod = []
train_data_eff = []
test_data_eff = []
train_data_cluster = []
test_data_cluster = []
for i, x in enumerate(prepr_data['X']):
    matrix = mp.preprocess_matrix(x)
    mod_score = bct.modularity_und(matrix)[1]
    eff_score = bct.efficiency_wei(matrix)
    local_clustering_coefficients = bct.clustering_coef_wu(matrix)
    
    if i in train_indices:
        train_data_comb.append([mod_score, eff_score])
        train_data_mod.append([mod_score])
        train_data_eff.append([eff_score])
        train_data_cluster.append(local_clustering_coefficients.tolist())
    else:
        test_data_comb.append([mod_score, eff_score])
        test_data_mod.append([mod_score])
        test_data_eff.append([eff_score])
        test_data_cluster.append(local_clustering_coefficients.tolist())


lr_preds = fit_and_pred_lr(train_data, train_label, test_data)
lr_comb_preds = fit_and_pred_lr(train_data_comb, train_label, test_data_comb)
lr_mod_preds = fit_and_pred_lr(train_data_mod, train_label, test_data_mod)
lr_eff_preds = fit_and_pred_lr(train_data_eff, train_label, test_data_eff)
lr_cluster_preds = fit_and_pred_lr(train_data_cluster, train_label, test_data_cluster)

svr = svm.SVR(C = 100, kernel = "linear", epsilon = 0.01)
svr_preds = svr.fit(train_data, train_label).predict(test_data)
svr_comb_preds = svr.fit(train_data_comb, train_label).predict(test_data_comb)
svr_mod_preds = svr.fit(train_data_mod, train_label).predict(test_data_mod)
svr_eff_preds = svr.fit(train_data_eff, train_label).predict(test_data_eff)
svr_cluster_preds = svr.fit(train_data_cluster, train_label).predict(test_data_cluster)

train_mean = mean(train_label)
mean_pred = [train_mean for _ in test_label]

print("guess: \t\t\t", metrics.mean_squared_error(test_label, mean_pred))
print("lr with whole matrices: ", metrics.mean_squared_error(test_label, lr_preds))
print("lr with mod and eff: \t", metrics.mean_squared_error(test_label, lr_comb_preds))
print("lr with mod: \t\t", metrics.mean_squared_error(test_label, lr_mod_preds))
print("lr with eff: \t\t", metrics.mean_squared_error(test_label, lr_eff_preds))
print("lr with clustering: \t", metrics.mean_squared_error(test_label, lr_cluster_preds))
print("svr with whole matrices:", metrics.mean_squared_error(test_label, svr_preds))
print("svr with mod and eff: \t", metrics.mean_squared_error(test_label, svr_comb_preds))
print("svr with mod: \t\t", metrics.mean_squared_error(test_label, svr_mod_preds))
print("svr with eff: \t\t", metrics.mean_squared_error(test_label, svr_eff_preds))
print("svr with clustering:\t", metrics.mean_squared_error(test_label, svr_cluster_preds))

print()
print("mean =", train_mean)

for idx, label in enumerate(test_label):
    print(label, lr_preds[idx], lr_comb_preds[idx], lr_mod_preds[idx], lr_eff_preds[idx],
        svr_preds[idx], svr_comb_preds[idx], svr_mod_preds[idx], svr_eff_preds[idx])
'''

# transformed = pca.pca_transform(mp.normalize_matrix(X))
# vis.visualize_PCA(transformed, y)

# transformed = pca.pca_transform(X)
# vis.visualize_PCA(transformed, y)


# lda = LDA(n_components=2) #2-dimensional LDA
# lda_transformed = pd.DataFrame(lda.fit_transform(X, y_df))

# plt.scatter(lda_transformed[y_df < -0.3][0], lda_transformed[y_df < -0.3][1], label='y < -0.3', c='red')
# plt.scatter(lda_transformed[(y_df >= -0.3) & (y_df <= 0.3)][0], lda_transformed[(y_df >= -0.3) & (y_df <= 0.3)][1], label='-0.3 <= y <= 0.3', c='blue')
# plt.scatter(lda_transformed[y_df > 0.3][0], lda_transformed[y_df > 0.3][1], label='y > 0.3', c='green')
# plt.legend(loc=3)
# plt.show()


# colors = [(0, 1, 0), (0, 0.5, 0.5), (0, 0, 1)]
# cm = LinearSegmentedColormap.from_list("asd", colors, N=100)

# # perps = [5, 10, 20, 30, 50, 60, 100, 110, 120, 150, 200]
# # perps = [2, 6, 10, 14, 18, 22]
# for i in range(1,6):
#   X_embedded = TSNE(n_components=2, perplexity=i, n_iter = 40000, 
#       init='random', n_iter_without_progress = 1000).fit_transform(X)

#   # vis.visualize_3d(X_embedded, y)

#   plt.scatter(X_embedded[:,0], X_embedded[:,1], cmap = cm, c = y)
#   plt.colorbar()
#   # plt.show()
#   plt.savefig("TSNE_perplexity_" + str(i) + ".pdf")
#   plt.close()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(X_embedded[:,0], X_embedded[:,1], y, c=kmeans.labels_)

# ax.set_xlabel('first')
# ax.set_ylabel('second')
# ax.set_zlabel('y')
# ax.legend()

# plt.scatter(X_embedded[:,0], X_embedded[:,1] c = kmeans.labels_)
# plt.colorbar()
# plt.show()