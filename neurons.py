inputs = [1, 2, 3, 2.5]
weights = [[0.3, 0.1, -0.5, 1.0],
           [0.4, -0.91, 0.25, -0.5],
           [-0.26, -0.21, 0.17, 0.87]]
biases = [2, 3, 0.5]

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print(layer_outputs)


# Inputs could be layers
#
# either true inputs from sensors or data or inputs from other hidden layer nuerons