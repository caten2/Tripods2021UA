from discrete_neural_net import Neuron, Layer, NeuralNet
from polymorphisms import RotationAutomorphism, IndicatorPolymorphism, polymorphism_neighbor_func, hamming_loss
from mnist_training_binary import binary_mnist_for_zero

training_pairs = tuple(binary_mnist_for_zero('train', 100))
constant_relations = tuple(pair[0]['x'] for pair in training_pairs)

layer0 = Layer(('x',))

neuron10 = Neuron(RotationAutomorphism(0), ('x',))
neuron11 = Neuron(RotationAutomorphism(0), ('x',))
layer1 = Layer([neuron10, neuron11])

neuron20 = Neuron(IndicatorPolymorphism((0, 0), (constant_relations[0], constant_relations[0])), [neuron10, neuron11])
layer2 = Layer([neuron20])

net = NeuralNet([layer0, layer1, layer2])

print(net.empirical_loss(training_pairs, hamming_loss))
print()

net.train(training_pairs, lambda op: polymorphism_neighbor_func(op, 4, constant_relations),
          100, hamming_loss, report_loss=True)
print()

print(net.empirical_loss(binary_mnist_for_zero('test', 100), hamming_loss))
