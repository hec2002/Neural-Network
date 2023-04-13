import numpy as np


class Dense:
    def __init__(self, input_size, output_size) -> None:
        # input = number of neurons in the input
        # output = number of neurons in the output
        # randomize weights to start
        self.weights = np.random.rand(output_size, input_size)
        self.bias = np.random.rand(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        # update parameters and return input gradient
        # calculate the derivative of the error with respect to the weights
        weights_gradient = np.dot(output_gradient, self.input.T)
        # adjust weigths according to gradient
        self.weights -= learning_rate * weights_gradient
        # adjust bias according to gradient
        self.bias -= learning_rate * output_gradient
        # return derivative of the error with respect to the input
        return np.dot(self.weights.T, output_gradient)
