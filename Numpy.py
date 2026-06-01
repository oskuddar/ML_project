''' oskuddar - Numphy Usage Example '''

import code

import numpy as np
seed = 49

# usage: rng.random(size=(h,w)) 
rng = np.random.default_rng(seed=seed)

'''Exercise 1:'''
# tanh = take all real numbers (-∞, ∞) and squish them into the range (-1, 1)
# tanh passes through the origin (0, 0) and is symmetric around it.
# tanh formula is sinh(x) / cosh(x) = (e^x - e^-x) / (e^x + e^-x)
# tanh (-x) = -tanh(x) (odd function)
# derivative of tanh is 1 - tanh^2(x)
# np.matmul() vs np.dot():- np.matmul() is a more general function that can handle higher-dimensional arrays and performs matrix multiplication according to the rules of linear algebra, while np.dot() is a more specific function that can perform both matrix multiplication and dot product depending on the input shapes. In practice, for 2D arrays, np.dot() and np.matmul() will yield the same result, but np.matmul() is recommended for clarity when performing matrix multiplication.

# Description of neural_network function: It takes two arguments, inputs and weights, which are expected to be 2D arrays (matrices). The function performs a matrix multiplication of the transposed weights with the inputs using np.matmul(), and then applies the hyperbolic tangent (tanh) activation function to the result. The output is a transformed version of the input data, where the values are squished into the range (-1, 1) due to the tanh function. This is a common operation in neural networks to introduce non-linearity into the model.  

def neural_network(inputs, weights):
    return np.tanh(np.matmul(weights.transpose(), inputs)) 
    # return np.tanh(weights.transpose() @ inputs) # alternative using the @ operator for matrix multiplication

inputs = rng.random(size=(2,1))
weights = rng.random(size=(2,1))

neural_network(inputs, weights)

'''Exercise 2:'''

# Scalar function

def scalar_function(x,y):
    if x <= y:
        return x * y
    else:
        return x / y
    
# Vectorized solution using np.where

def scalar_function (x,y):
    return np.where(x<=y, x*y, x/y) # if x <= y, return x*y, else return x/y

# Vector function that uses scalar_function 

def vector_function(x,y):
    return np.vectorize(scalar_function)(x,y)

