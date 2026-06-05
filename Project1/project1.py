# Implement and compare three types of linear classifiers: the perceptron algorithm, the average perceptron algorithm, and the Pegasos algorithm.

# First, implement the basic hinge loss calculation on a single data-point. Instead of the entire feature matrix, you are given one row, representing the feature vector of a single data sample, and its label of +1 or -1 representing the ground truth sentiment of the data sample.

import numpy as np
import pandas as pd

toy_data = pd.read_csv("./Project1/sentiment_analysis/toy_data.tsv", sep='\t', encoding="latin-1")

# toy_data.head()
# toy_data.columns.tolist()
feature_vector = np.asarray(toy_data.iloc[:,1:3], dtype=float)
label = np.asarray(toy_data.iloc[:,0], dtype=float)
theta =np.asarray([0,0], dtype=float)
theta_0 = 0.0

def hinge_loss_single(feature_vector, label, theta, theta_0):
    for i in range(len(feature_vector)):
        db = label[i] * ((theta @ feature_vector[i]) + theta_0)
        print(db)
        h_loss = max (0.0, 1.0 - db)
        print(h_loss)
        if h_loss > 0.0:
            theta += np.dot(label[i],feature_vector[i])
            theta_0 += feature_vector[i]
            print(theta, theta_0)        
        
hinge_loss_single(feature_vector, label, theta, theta_0)
