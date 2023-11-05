"""
Nullary operations as weights
"""

from neural_net import Neuron, Layer, NeuralNet
from polymorphisms import RotationAutomorphism, IndicatorPolymorphism, polymorphism_neighbor_func, hamming_loss
from mnist_training_binary import binary_mnist_zero_one
from relations import sliced_random_atomic_relations
from operations import Constant

training_pairs = tuple(binary_mnist_zero_one(100, 'train'))
constant_relations = sliced_random_atomic_relations(28, 2, 100)

layer0 = Layer(('x0',))

neuron10 = Neuron(Constant(constant_relations[0]), tuple())
neuron11 = Neuron(RotationAutomorphism(0), ('x0',))
layer1 = Layer([neuron10, neuron11])

neuron20 = Neuron(IndicatorPolymorphism((0, 0), (constant_relations[0], constant_relations[0])), [neuron10, neuron11])
layer2 = Layer([neuron20])

net = NeuralNet([layer0, layer1, layer2])

print(net.empirical_loss(training_pairs, hamming_loss))
print()

net.train(training_pairs, lambda op: polymorphism_neighbor_func(op, 4, constant_relations),
          100, hamming_loss, report_loss=True)
print()

