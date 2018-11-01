import numpy as np

Z = np.array([[1,2,3],[4,5,6],[7,8,9],[1,3,5],[2,4,6],[3,5,7]])

r = np.sum(Z, axis=1)

print(np.nonzero(r > 10))
