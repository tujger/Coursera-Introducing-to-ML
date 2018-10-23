import numpy as np
import pandas
from sklearn.metrics import roc_auc_score

np.set_printoptions(linewidth=120, threshold=np.inf)

data = pandas.read_csv('data/data-logistic.csv', names=['T', 'A', 'B'], header=None)

X = data.values[:, 1:]
y = data.values[:, :1].T[0]

# roc = roc_auc_score(train.T, scores)

def euklid(a, b):
    return np.sqrt(np.square(a[0] - b[0]) + np.square(a[1] - b[1]))


def sigmoid(X, i, w):
    return 1 / (1 + np.exp(-w[0] * X[i,0] - w[1] * X[i,1]))


def log_regression(X_, y_, k, w, C, epsilon, max_iter):
    w1, w2 = w
    for i in range(max_iter):
        w1new = w1 + k * np.mean(y_ * X_[:, 0] * (1. - 1. / (1. + np.exp(-y_ * (w1 * X_[:, 0] + w2 * X_[:, 1]))))) - k * C * w1
        w2new = w2 + k * np.mean(y_ * X_[:, 1] * (1. - 1. / (1. + np.exp(-y_ * (w1 * X_[:, 0] + w2 * X_[:, 1]))))) - k * C * w2
        if euklid((w1new, w2new), (w1, w2)) < epsilon:
            break
        w1, w2 = w1new, w2new

    print(w1,w2)
    return 1.0 / (1.0 + np.exp(-X_.dot([w1,w2])))


notreg = log_regression(X, y, 0.1, [.0,.0], 0, 1e-5, 10000)

# notreg_scores = sigmoid(X.values, notreg)

print("Not regularized:",round(roc_auc_score(y, notreg),3))

reg = log_regression(X, y, 0.1, [.0,.0], 10, 1e-5, 10000)
# reg_scores = sigmoid(X.values, reg)

print("Regularized with C=10:",round(roc_auc_score(y, reg),3))



# np.gradient()
#
# print(calc_regr(1,2))
# print(calc_regr(3,4))




