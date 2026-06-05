# Implement and compare three types of linear classifiers: the perceptron algorithm, the average perceptron algorithm, and the Pegasos algorithm.

# First, implement the basic hinge loss calculation on a single data-point. Instead of the entire feature matrix, you are given one row, representing the feature vector of a single data sample, and its label of +1 or -1 representing the ground truth sentiment of the data sample.

import numpy as np
import pandas as pd

review_train = pd.read_csv("./Project1/sentiment_analysis/reviews_train.tsv", sep='\t', encoding="latin-1")

review_train.head()
review_train.columns.tolist()
feature_vector = np.asarray(review_train['sentiment'])
np.shape(feature_vector)

def hinge_loss_single(feature_vector, label, theta, theta_0):
    