import numpy as np
import pandas as pd

toy_data = pd.read_csv("./Project1/sentiment_analysis/toy_data.tsv", sep='\t', encoding="latin-1")

# toy_data.head()
# toy_data.columns.tolist()
feature_matrix = np.asarray(toy_data.iloc[:,1:3], dtype=float)
label = np.asarray(toy_data.iloc[:,0], dtype=float)
theta =np.asarray([0,0], dtype=float)
theta_0 = 0.0
minval = 1e-7
T = 1

def perceptron_single_step_update (feature_vector, label, theta, theta_0):
        db = label * ((feature_vector @ theta) + theta_0)
        if db <= minval:
            theta += np.dot(label,feature_vector)
            theta_0 += label
        return (theta, theta_0)     

# def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
#     if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 1e-7:
#         return (current_theta + label * feature_vector, current_theta_0 + label)
#     return (current_theta, current_theta_0)

def perceptron (feature_matrix, label, T):
    theta,theta_0 = np.zeros(feature_matrix.shape[1]), 0.0
    for t in range(T):
        #for i in get_order(feature_matrix.shape[0])
        for i in range(len(feature_matrix)):
            theta, theta_0 = perceptron_single_step_update (feature_matrix[i], label[i], theta, theta_0)
    return print(theta, theta_0)


perceptron (feature_matrix, label, T)

# def perceptron(feature_matrix, labels, T):
#     (nsamples, nfeatures) = feature_matrix.shape
#     theta = np.zeros(nfeatures)
#     theta_0 = 0.0
#     for t in range(T):
#         for i in get_order(nsamples):
#             theta, theta_0 = perceptron_single_step_update(
#                 feature_matrix[i], labels[i], theta, theta_0)
#     return (theta, theta_0)



