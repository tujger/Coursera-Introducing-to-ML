import numpy as np

np.set_printoptions(linewidth=250, threshold=np.inf)

X = np.random.normal(loc=1, scale=10, size=(1000,50))

print(X)

print("========================")

m = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = (X-m)/std

print(X_norm)
