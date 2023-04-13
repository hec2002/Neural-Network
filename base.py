class Layer():
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        # returns output
        pass

    def back_propagation(self, output_gradient, learning_rate):
        # update parameters and return input gradient
        pass
