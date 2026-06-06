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

def perceptron_single_step_update (feature_vector, label, theta, theta_0):
        db = label * ((feature_matrix @ theta) + theta_0)
        print(db)
        if db <= minval:
            theta += np.dot(label,feature_matrix)
            theta_0 += label
        return (theta, theta_0)        
    
perceptron_single_step_update (feature_vector, label, theta, theta_0)

