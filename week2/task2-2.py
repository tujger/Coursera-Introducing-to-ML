import numpy as np
import pandas
import sklearn
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston

np.set_printoptions(linewidth=120, threshold=np.inf)


data = load_boston()

features = scale(data.data)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

x_max = -100
i_max = 0

for x in np.linspace(1,10,200):
    clf = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=x)
    scores = cross_val_score(clf, features, data.target, cv=kfold, scoring='neg_mean_squared_error')
    mean = np.mean(scores)
    if x_max < mean:
        x_max = mean
        i_max = x
    print('p: %s, scores: %s, mean: %s' % (x, scores, mean.round(3)))

print('Max: ', x_max.round(2), ', p: ', i_max.round(2))

