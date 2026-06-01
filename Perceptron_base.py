
# Perceptron Algorithm

import numpy as np


def perceptron (n_iter, x=None, y=None):
    theta = np.array([0,0], dtype=float)
    # x = np.asarray([] if x is None else x)
    # y = np.asarray([] if y is None else y)
    for i in range(n_iter):
        for j in range(len(x)):
            if (y[j] * (theta @ x[j].T)) <= 0:
                theta += np.dot (y[j], x[j])
            print("j= ", j, " ", theta)



x = np.array([[-1, -1], [1, 0], [-1, 1.5]])
y = np.array([1, -1, 1])

# Change Order
# x = np.array([[1, 0], [-1, 1.5], [-1, -1]])
# y = np.array([-1, 1, 1])
n_iter = 5
perceptron (n_iter, x, y)
