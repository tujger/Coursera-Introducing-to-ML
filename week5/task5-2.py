# Градиентный бустинг над решающими деревьями

import numpy as np
import pandas
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


np.set_printoptions(linewidth=120, threshold=np.inf)

data = pandas.read_csv('data/gbm-data.csv')

y = data['Activity'].values
X = data.loc[:, 'D1':].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

for n in [1, 0.5, 0.3, 0.2, 0.1]:
    clf = GradientBoostingClassifier(n_estimators=250, learning_rate=n, verbose=False, random_state=241)
    clf.fit(X_train, y_train)
    print('N:', n)

    train_scores = []
    test_scores = []
    train_scores1 = []
    test_scores1 = []

    min_train_logloss = 1e6
    min_train_i = 0
    for i, y_pred in enumerate(clf.staged_decision_function(X_train)):
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        logloss = log_loss(y_train, y_pred)
        train_scores.append(logloss)
        if logloss < min_train_logloss:
            min_train_logloss = logloss
            min_train_i = i

    min_test_logloss = 1e6
    min_test_i = 0
    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        logloss = log_loss(y_test, y_pred)
        test_scores.append(logloss)
        if logloss < min_test_logloss:
            min_test_logloss = logloss
            min_test_i = i

    print('Min train:', np.round(min_train_logloss, 2), ', i:', min_train_i)
    print('Min test:', np.round(min_test_logloss, 2), ', i:', min_test_i)

    plt.figure(n)
    plt.title(n)
    plt.plot(train_scores, 'r', linewidth=2)
    plt.plot(test_scores, 'b', linewidth=2)
    plt.legend(['train', 'test', 'loss_'])
    plt.show()

    if min_test_i > 0:
        clf = RandomForestClassifier(n_estimators=min_test_i, random_state=241)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)
        logloss = log_loss(y_test, y_pred)
        print('RF score:', np.round(logloss, 2))


