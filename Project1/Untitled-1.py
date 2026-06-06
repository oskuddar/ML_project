
def hinge_loss(feature_matrix, label, theta, theta_0):
    total = 0.0
    for i in range(len(feature_matrix)):
        db = label[i] * ((feature_matrix[i] @ theta) + theta_0)
        h_loss = np.maximum (0.0, 1.0 - db)
        if h_loss > 0.0:
            theta += np.dot(label[i],feature_matrix[i])
            theta_0 += label[i]
        total += h_loss
    print(total / len(feature_matrix))
    return np.mean(h_loss)
    
hinge_loss(feature_matrix, label, theta, theta_0)
