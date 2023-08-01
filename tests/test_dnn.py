"""
Discrete neural net test
"""

from discrete_neural_net import Neuron, Layer, NeuralNet
import operations
from itertools import product

# Our neural net will have three inputs.
layer0 = Layer(('x0', 'x1', 'x2'))

# The neural net will have input and output values which are integers modulo `order`.
order = 100

# The first layer has two neurons, which are initialized to carry modular addition and a random operation as
# activation functions.
neuron0 = Neuron(operations.ModularAddition(order), ('x0', 'x1'))
neuron1 = Neuron(operations.RandomOperation(order, 2), ('x1', 'x2'))
layer1 = Layer([neuron0, neuron1])

# The third layer has a single neuron, which is initialized to carry modular multiplication.
neuron2 = Neuron(operations.ModularMultiplication(5), [neuron0, neuron1])
layer2 = Layer([neuron2])

net = NeuralNet([layer0, layer1, layer2])

# We can feed values forward and display the result.
print(net.feed_forward({'x0': 0, 'x1': 1, 'x2': 2}))
print()

# We create a training set in an effort to teach our net how to compute (x0+x1)*(x1+x2).
# We'll do this modulo `order`.
training_pairs = [({'x0': x[0], 'x1': x[1], 'x2': x[2]}, (((x[0] + x[1]) * (x[1] + x[2])) % order,))
                  for x in product(range(order // 2 + 1), repeat=3)]

# We can check out empirical loss with respect to this training set.
# Our loss function will just be the 0-1 loss.
print(net.empirical_loss(training_pairs))
print()


def neighbor_func(op):
    """
    Report all the neighbors of any operation as being addition, multiplication, or a random binary operation.
    Our example only has binary operations for activation functions so we don't need to be any more detailed than this.

    Argument:
        op (operation): The Operation whose neighbors we'd like to find.

    Returns:
        list of Operations: The neighboring Operations.
    """

    return [operations.ModularAddition(order),
            operations.ModularMultiplication(order),
            operations.RandomOperation(order, 2)]


# We can now begin training.
# Usually it will only take a few training steps to learn to replace the random operation with addition.
net.train(training_pairs, neighbor_func, 5, report_loss=True)
