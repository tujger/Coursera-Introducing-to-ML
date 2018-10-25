# Размер случайного леса

import numpy as np
import pandas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


np.set_printoptions(linewidth=120, threshold=np.inf)

data = pandas.read_csv('data/gbm-data.csv')

y = data['Activity'].values
X = data.loc[:, 'D1':].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

def sigmoid(X, i, w):
    return 1 / (1 + np.exp(-w[0] * X[i,0] - w[1] * X[i,1]))
    return 1.0 / (1.0 + np.exp(-pred))


for n in [1]:#, 0.5, 0.3, 0.2, 0.1]:
    clf = GradientBoostingClassifier(n_estimators=250, learning_rate=n, verbose=True, random_state=241)
    clf.fit(X_train, y_train)
    sdf_train = clf.staged_decision_function(X_train)
    sdf_test = clf.staged_decision_function(X_test)
    print('N:', n, ', train:', sdf_train, ', test:', sdf_test)

    train_scores = []
    train_scores1 = []
    test_scores = []
    test_scores1 = []
    for i, pred in enumerate(clf.staged_decision_function(X_test)):
        # a = clf.score(pred, y_test)
        test_scores.append(clf.loss_(y_test, pred))
        b = 1.0 / (1.0 + np.exp(-pred))
        test_scores1.append(log_loss(y_test, b))
        # print(log_loss(y_test, b), clf.loss_(y_test, pred))

    for i, pred in enumerate(clf.staged_decision_function(X_train)):
        train_scores.append(clf.loss_(y_train, pred))
        b = 1.0 / (1.0 + np.exp(-pred))
        train_scores1.append(log_loss(y_train, b))
        # print(log_loss(y_train, b), clf.loss_(y_train, pred))


    plt.figure(n)
    plt.plot(train_scores, 'r', linewidth=2)
    plt.plot(test_scores, 'g', linewidth=2)
    plt.legend(['train', 'test'])
    plt.show()

    plt.figure(n + 1)
    plt.plot(train_scores1, 'r', linewidth=2)
    plt.plot(test_scores1, 'g', linewidth=2)
    plt.legend(['train1', 'test1'])
    plt.show()
