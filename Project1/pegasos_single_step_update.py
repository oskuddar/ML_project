# PEGASOS algorithm
# eta_param = decaying factor that will decrease over time.
# lambda_param = regularizing parameter

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
T = 1
lambda_param = 0.1
eta_param = 0.1

def pegasos_single_step_update (x_vector, label, lambda_param, eta_param, theta, theta_0):
    if label * ((theta.T @ x_vector) + theta_0)  <= 1:
        theta += (eta_param * ((label * x_vector) - (lambda_param * theta)))
        theta_0 += eta_param * label
    else:
        theta += -(eta_param * lambda_param) * theta
    return (theta, theta_0)

pegasos_single_step_update (feature_matrix[4], label[4], lambda_param, eta_param, theta, theta_0)
