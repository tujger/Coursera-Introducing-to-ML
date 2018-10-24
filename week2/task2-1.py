# Выбор числа соседей

import numpy as np
import pandas
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

np.set_printoptions(linewidth=120, threshold=np.inf)

names = ['Class', 'Alcohol', 'MalicAcid', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols', 'Flavanoids',
         'Nonflavanoid', 'Proanthocyanins', 'ColorIntensity', 'Hue', 'OD', 'Proline']

data = pandas.read_csv('data/wine.data', names=names)

y_digits = data.Class
X_digits = data[['Alcohol', 'MalicAcid', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols', 'Flavanoids',
                'Nonflavanoid', 'Proanthocyanins', 'ColorIntensity', 'Hue', 'OD', 'Proline']]


kfold = KFold(n_splits=5, shuffle=True, random_state=42)
clf = KNeighborsClassifier(n_neighbors=1)

x_max = 0
i_max = 0

for x in range(1,51):
    clf = KNeighborsClassifier(n_neighbors=x)
    scores = cross_val_score(clf, X_digits, y_digits, cv=kfold, scoring='accuracy')
    mean = np.mean(scores)
    if x_max < mean:
        x_max = mean
        i_max = x
    # print('K: %s, scores: %s, mean: %s' % (x, scores, mean.round(3)))

print('Max: ', x_max.round(2), ', K: ', i_max)

X_digits = scale(X_digits)

x_max = 0
i_max = 0

for x in range(1,51):
    clf = KNeighborsClassifier(n_neighbors=x)
    scores = cross_val_score(clf, X_digits, y_digits, cv=kfold, scoring='accuracy')
    mean = np.mean(scores)
    if x_max < mean:
        x_max = mean
        i_max = x
    print('K: %s, scores: %s, mean: %s' % (x, scores, mean.round(3)))

print('Scaled max: ', x_max.round(2), ', K: ', i_max)
