import numpy as np
import pandas
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.set_printoptions(linewidth=120, threshold=np.inf)

train = pandas.read_csv('data/svm-data.csv', names=['T', 'A', 'B'])

X_train = np.array(train[['A','B']])
y_train = np.array(train['T'])

clf = SVC(kernel='linear', C=100000, random_state=241)

clf.fit(X_train, y_train)

print(clf.support_)
