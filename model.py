# modular implementation design allows for ease in model creation
from cmath import tanh
from re import X
from dense import Dense
from tanh import Tanh
from losses import mse, mse_prime
import numpy as np
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]
# hyperparameters
epochs = 1000
learning_rate = 0.1

# train
for e in range(epochs):
    error = 0
    # forward propagation
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)
        # error
        error += mse(y, output)
        # backpropagation
        gradient = mse_prime(y, output)
        for layer in reversed(network):
            gradient = layer.backward(gradient, learning_rate)
    error /= len(X)
    print('%d/%d, error=%f' % (e + 1, epochs, error))
