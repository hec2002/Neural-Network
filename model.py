def prediction(network, output):
    for layer in network:
        output = layer.forward(output)
    return output
def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, print_error = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = prediction(network, x)

            # error
            error += loss(y, output)

            # backward
            gradient = loss_prime(y, output)
            for layer in reversed(network):
                gradient = layer.backward(gradient, learning_rate)

        error /= len(x_train)
        if print_error:
            print(f"{e + 1}/{epochs}, error={error}")