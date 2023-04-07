"""
Discrete neural net test
"""

from discrete_neural_net import Operation, Neuron, Layer, NeuralNet
import operations

# Our neural net will have three inputs.
layer1 = Layer(('x1', 'x2', 'x3'))

neuron1 = Neuron(operations.ModularAddition(5), ('x1', 'x2'))
neuron2 = Neuron(operations.RandomOperation(5, 2), ('x2', 'x3'))
layer2 = Layer([neuron1, neuron2])

neuron3 = Neuron(operations.ModularAddition(5), [neuron1, neuron2])
layer3 = Layer([neuron3])

net = NeuralNet([layer1, layer2, layer3])

print(net.feed_forward({'x1': 0, 'x2': 1, 'x3': 2}))
