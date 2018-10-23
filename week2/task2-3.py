import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.set_printoptions(linewidth=120, threshold=np.inf)

train = pandas.read_csv('data/perceptron-train.csv', names=['T', 'A', 'B'])
test = pandas.read_csv('data/perceptron-test.csv', names=['T', 'A', 'B'])

X_train = np.array(train[['A','B']])
y_train = np.array(train['T'])

X_test = np.array(test[['A','B']])
y_test = np.array(test[['T']])

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print('Accuracy =',accuracy)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clfScaled = Perceptron(random_state=241)
clfScaled.fit(X_train_scaled, y_train)
predictionsScaled = clfScaled.predict(X_test_scaled)

accuracyScaled = accuracy_score(y_test, predictionsScaled)

print('Accuracy scaled =',accuracyScaled)

print('Accuracy delta =',round(accuracyScaled - accuracy, 3))
