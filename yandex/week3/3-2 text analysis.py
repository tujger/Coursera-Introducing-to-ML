# Анализ текстов

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer

np.set_printoptions(linewidth=120, threshold=np.inf)

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
feature_mapping = vectorizer.get_feature_names()

grid = {'C': np.power(10.0, np.arange(-5, 6))}
clf = SVC(kernel='linear', random_state=241)
kfold = KFold(n_splits=5, shuffle=True, random_state=241)

print('GridSearch started for', X.size, 'elements')

gs = GridSearchCV(clf, grid, scoring='accuracy', cv=kfold, n_jobs=-1)
gs.fit(X, newsgroups.target)

print('GridSearch finished')

scores = gs.best_estimator_.coef_
sorted_scores = np.argsort(np.abs(scores.toarray()[0]))[-10:]

biggest_words = [feature_mapping[index] for index in sorted_scores]

print(sorted(biggest_words))
