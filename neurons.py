import numpy as np


class Neuron:
    def __init__(self, inputs, weights):
        self.inputs = inputs
        self.weights = weights

    def single_output(self, a, b, bias):
        output = np.dot(a, b) + bias
        print(output)
        return output

# inputs = [1, 2, 3, 2.5]
# weights = [[0.3, 0.1, -0.5, 1.0],
#            [0.4, -0.91, 0.25, -0.5],
#            [-0.26, -0.21, 0.17, 0.87]]
# biases = [2, 3, 0.5]


first = Neuron([1, 2, 3, 2.5], [[0.3, 0.1, -0.5, 1.0],
                                [0.4, -0.91, 0.25, -0.5],
                                [-0.26, -0.21, 0.17, 0.87]])
first.single_output(first.weights, first.inputs, [1.2, 3, -0.4])
# print(output)


# layer_outputs = []
# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += n_input*weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)
# print(layer_outputs)


# Inputs could be layers
#
# either true inputs from sensors or data or inputs from other hidden layer nuerons
# Dot products in terms of numpy
