"""
Tests for dominion polymorphisms
"""
from neural_net import Neuron, Layer, NeuralNet
import arithmetic_operations
from random_neural_net import RandomOperation
from itertools import product
from dominion import *
from polymorphisms import RotationAutomorphism, ReflectionAutomorphism, SwappingAutomorphism, BlankingEndomorphism, \
    IndicatorPolymorphism

# We will train the NN to compute the function f(r_0, r_1) = r_0 * r_1 (as relations)
order = 100

def

# TODO
training_pairs = [({'x0': x[0], 'x1': x[1], 'x2': x[2]}, (((x[0] ** 2) + (x[1] * x[2])) % order,))
                  for x in product(range(order // 2), repeat=3)]

# The NN will have 3 inputs
layer0 = Layer(('x0', 'x1', 'x2'))

neuron0 = Neuron(RandomOperation(order, 2), ('x0', 'x1'))
neuron1 = Neuron(RandomOperation(order, 2), ('x1', 'x2'))
layer1 = Layer([neuron0, neuron1])

neuron2 = Neuron(arithmetic_operations.ModularAddition(100), [neuron0, neuron1])
layer2 = Layer([neuron2])

net = NeuralNet([layer0, layer1, layer2])

# We can feed values forward and display the result.
print(net.feed_forward({'x0': 0, 'x1': 1, 'x2': 2}))
print()

# We can check out empirical loss with respect to this training set.
# Our loss function will just be the 0-1 loss.
print(net.empirical_loss(training_pairs))
print()


# TODO
def neighbor_func(op):
    """
    Report all the neighbors of any operation as being addition, multiplication, or a random binary operation.
    Our example only has binary operations for activation functions so we don't need to be any more detailed than this.

    Argument:
        op (operation): The Operation whose neighbors we'd like to find.

    Returns:
        list of Operations: The neighboring Operations.
    """

    return [arithmetic_operations.ModularAddition(order),
            arithmetic_operations.ModularMultiplication(order),
            RandomOperation(order, 2)]


r1 = Relation([(0, 0)], 3, 2)
r2 = r1
r3 = Relation([], 3, 2)
a = random_dominion_polymorphism(3, range(5))
print(a(r1, r2))
print(a(r1, r3))


