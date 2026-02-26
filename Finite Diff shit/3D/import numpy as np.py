import numpy as np

A = np.eye(4) - np.matrix([[0, 0.09, 0.81, 0],  [0.81, 0.09, 0, 0],  [0, 0, 0.09, 0.81],  [0.81, 0, 0, 0.09]])
b = np.matrix([0,1,0,2]).T
v = np.linalg.solve(A,b)  # A v = b
print(v.tolist)
