"""
Train a discrete neural net using only endomorphisms
"""
from neural_net import Neuron, Layer, NeuralNet
from polymorphisms import RotationAutomorphism, polymorphism_neighbor_func, hamming_loss
from mnist_training_binary import binary_mnist_zero_one

# Our neural net will have one input.
layer0 = Layer(('x0',))

# There is only a single output neuron and no hidden layers.
neuron0 = Neuron(RotationAutomorphism(0), ('x0',))
layer1 = Layer([neuron0])

net = NeuralNet([layer0, layer1])

# The MNIST training set will be used to train this discrete neural net to detect the handwritten digit 0.
# Load some binary images from the modified MNIST training set.
# Note that `binary_mnist_zero_one` is a generator, so it must be made into a tuple in order to be reused.
training_pairs = tuple(binary_mnist_zero_one(10, 'train', 100))

# We can check out empirical loss with respect to this training set.
# Our loss function will be the Hamming distance, which is the size of the symmetric difference of two relations.
print(net.empirical_loss(training_pairs, hamming_loss))
print()

# Use the training set for a tuple of constant images to use for swapping/blanking endomorphisms.
constant_relations = tuple(pair[0]['x0'] for pair in training_pairs)

# Train the neural net using this training data.
net.train(training_pairs, lambda op: polymorphism_neighbor_func(op, 4, constant_relations),
          100, hamming_loss, report_loss=True)
print()

# We can also test the neural net we have trained against the MNIST test data.
print(net.empirical_loss(binary_mnist_zero_one(10, 'test', 100), hamming_loss))

# Let's see what the model has actually learned.
for pair in training_pairs:
    pair[0]['x0'].show('sparse')
    net.feed_forward(pair[0])[0].show('sparse')
    print()

# Most images in the training data are not handwritten zeroes, so our net has learned to guess that the answer is a
# mostly blank image. That is, it's basically always guessing the number will not be a handwritten 0.
