from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
def build_model(X, y):
    reg = MLPRegressor(hidden_layer_sizes=(25,), max_iter=200000000)
    model = reg.fit(X, y)
    return(model)
