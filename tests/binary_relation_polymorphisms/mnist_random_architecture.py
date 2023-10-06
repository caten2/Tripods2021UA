from random_neural_net import RandomNeuralNet
from polymorphisms import IndicatorPolymorphism, polymorphism_neighbor_func, hamming_loss
from mnist_training_binary import binary_mnist_zero_one
from random import choices

training_pairs = tuple(binary_mnist_zero_one(100, 'train'))
constant_relations = tuple(pair[0]['x0'] for pair in training_pairs)

# There is some issue with the higher-arity Indicator polymorphisms here, so really we're just using a single binary
# one for our initial activation functions.
net = RandomNeuralNet(
    {n: [IndicatorPolymorphism(tuple(0 for _ in range(n)), choices(constant_relations, k=n))] for n in range(2, 3)},
    ('x0',), 1, 4, 4)

print(net.empirical_loss(training_pairs, hamming_loss))
print()

net.train(training_pairs, lambda op: polymorphism_neighbor_func(op, 4, constant_relations),
          100, hamming_loss, report_loss=True)
print()
