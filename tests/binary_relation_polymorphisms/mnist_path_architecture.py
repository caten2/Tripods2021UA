"""
Train a discrete neural net whose architecture is a path
"""
from discrete_neural_net import Neuron, Layer, NeuralNet
from polymorphisms import RotationAutomorphism, polymorphism_neighbor_func, hamming_loss
from mnist_training_binary import binary_mnist_for_zero

num_of_layers = 100
neurons = [('x',)]
for _ in range(1, num_of_layers):
    neurons.append([Neuron(RotationAutomorphism(0), neurons[-1])])
layers = [Layer(neuron_lis) for neuron_lis in neurons]

net = NeuralNet(layers)

training_pairs = tuple(binary_mnist_for_zero('train', 100))

print(net.empirical_loss(training_pairs, hamming_loss))
print()

constant_relations = tuple(pair[0]['x'] for pair in training_pairs)

net.train(training_pairs, lambda op: polymorphism_neighbor_func(op, 10, constant_relations),
          100, hamming_loss, report_loss=True)
print()

print(net.empirical_loss(binary_mnist_for_zero('test', 100), hamming_loss))
