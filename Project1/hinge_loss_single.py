# Implement and compare three types of linear classifiers: the perceptron algorithm, the average perceptron algorithm, and the Pegasos algorithm.

# First, implement the basic hinge loss calculation on a single data-point. Instead of the entire feature matrix, you are given one row, representing the feature vector of a single data sample, and its label of +1 or -1 representing the ground truth sentiment of the data sample.


import numpy as np
import pandas as pd

def hinge_loss_single(feature_vector, label, theta, theta_0):
        db = label[i] * ((feature_vector[i] @ theta) + theta_0) # @=dot product
        h_loss = max (0.0, 1.0 - db)
        return h_loss
        
