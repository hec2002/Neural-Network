from cmath import tanh
from re import X
from dense import Dense
from tanh import Tanh
from losses import mse, mse_prime
import numpy as np
from model import train

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
train(network, mse, mse_prime, X, Y)
