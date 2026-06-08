
import pandas as pd
import numpy as np

toy_data = pd.read_csv("/Users/shera/VSCode/ML_project/Project1/sentiment_analysis/toy_data.tsv", sep='\t', encoding="latin-1")

# toy_data.head()
# toy_data.columns.tolist()
feature_matrix = np.asarray(toy_data.iloc[:,1:3], dtype=float)
label = np.asarray(toy_data.iloc[:,0], dtype=float)
theta =np.asarray([0,0], dtype=float)
theta_0 = 0.0
minval = 1e-7
T = 20
lambda_param = 0.2

# def get_order(matrix):
#     return len(matrix)

def pegasos_single_step_update (x_vector, label, lambda_param, eta_param, theta, theta_0):
    if label * ((theta.T @ x_vector) + theta_0)  <= 1:
        theta += (eta_param * ((label * x_vector) - (lambda_param * theta)))
        theta_0 += eta_param * label
    else:
        theta += -(eta_param * lambda_param) * theta
    return (theta, theta_0)

def pegasos(feature_matrix, label, T, lambda_param):
    theta =np.asarray([0,0], dtype=float)
    theta_0 = 0.0
    count = 0
    for t in range(1,T+1):
        for i in range(len(feature_matrix)):
            count+=1
            eta_param = 1/np.sqrt(count)
            theta, theta_0 = pegasos_single_step_update (feature_matrix[i], label[i], lambda_param, eta_param, theta, theta_0)
    return print(theta, theta_0)

pegasos(feature_matrix, label, T, lambda_param)

