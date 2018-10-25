# Размер случайного леса

import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score


np.set_printoptions(linewidth=120, threshold=np.inf)

data = pandas.read_csv('data/abalone.csv')

# data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
data['Sex'].replace({'F': -1, 'I': 0, 'M': 1}, inplace=True)

# X = np.array(data[['Sex','Length','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight']])
# y = np.array(data[['Rings']])


X = data.loc[:,'Sex':'ShellWeight']
y = data['Rings']

# y['Sex'] = y['Sex'] == 'F'

# Sex,Length,Diameter,Height,WholeWeight,ShuckedWeight,VisceraWeight,ShellWeight,Rings

kfold = KFold(random_state=1, shuffle=True, n_splits=5)
total_scores = [0.0]
for i in range(1,50):
    clf = RandomForestRegressor(n_estimators=i, random_state=1)

    # scores = cross_val_score(clf, X, y.ravel(), scoring='r2', cv=kfold)
    # mean = np.average(scores)
    # min = np.min(scores)
    # max = np.max(scores)

    # print('NC:', i, ', min:', min.round(4), ', max:', max.round(4), ', mean:', mean.round(4), ', scores:', scores)

    scores = []
    sc = 0
    for train, test in kfold.split(X):
        clf = RandomForestRegressor(n_estimators=i, random_state=1)
        clf.fit(X.loc[train], y.loc[train])
        y_predict = clf.predict(X.loc[test])
        score = r2_score(y.loc[test], y_predict)

        scores.append(score)
    total_scores.append(np.mean(scores))
    print('N:', i, ', min:', np.round(np.min(scores), 3), ', max:', np.round(np.max(scores), 3), ', mean:', np.round(np.mean(scores), 3), ', scores:', scores)

for n, score in enumerate(total_scores):
    if score > 0.52:
        print(n)
        break