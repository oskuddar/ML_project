
# Perceptron Algorithm

import numpy as np


def perceptron (n_iter, x=None, y=None):
    theta = np.array([0,0], dtype=float)
    theta_not = 0.0
    # x = np.asarray([] if x is None else x)
    # y = np.asarray([] if y is None else y)
    for i in range(n_iter):
        for j in range(len(x)):
            if (y[j] * ((theta @ x[j].T) + theta_not)) <= 0:
                theta += np.dot (y[j], x[j])
                theta_not += y[j]
                print("mistake:", "epoch=", i, "j=", j, "theta=", theta, "theta_not=", theta_not)
    print(theta)
    print(theta_not)



x = np.array([[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]])
y = np.array([1, 1, -1, -1, -1])

n_iter = 10
perceptron (n_iter, x, y)
