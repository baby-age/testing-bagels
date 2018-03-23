from sklearn.svm import SVR
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import sklearn.neural_network

def build_model(X, y):
    reg = sklearn.neural_networ.MLPRegressor(hidden_layer_sizes=(25,), max_iter=200000000)
    model = reg.fit(X, y)
    return(model)

def predict_and_score(X_train, y_train, X_test, y_test, model = LinearRegression()):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions, metrics.mean_squared_error(y_test, predictions)

'''
Maybe not the right place for this function
'''
def tune_SVR_params(X_train, y_train, X_test, y_test):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000], 
                         'epsilon': [3, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]},
                        {'kernel': ['linear'], 'C': [1, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000], 
                        'epsilon': [3, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]}]

    scores = ['neg_mean_absolute_error', 'explained_variance']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        svr = GridSearchCV(SVR(), tuned_parameters, cv = 15, scoring = score)
        svr.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(svr.best_params_)
        
        y_true, y_pred = y_test, svr.predict(X_test)
        print("mean absolute error:", metrics.mean_absolute_error(y_true, y_pred))
        print("explained variance", metrics.explained_variance_score(y_true, y_pred))
        print()
