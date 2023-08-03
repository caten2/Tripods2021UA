"""
Train a discrete neural net
"""
from pathlib import Path

path = str(Path(__file__).parent.parent.absolute() / "src")
import sys
myFolderPath = '/Users/kevinxue/Downloads/Tripods2023/Tripods2021UA/src'
sys.path.insert(0, path)

from discrete_neural_net import Neuron, Layer, NeuralNet
from binary_image_polymorphisms import RotationAutomorphism, polymorphism_neighbor_func, hamming_distance
from mnist_training_binary import binary_train_for_zero, show
from dominion import getGAlpha


# Our neural net will have one input.
layer0 = Layer(('x',))

# Create gAlpha
gAlpha=getGAlpha(3)


# The first layer has two neurons which are initialized to carry out rotations
neuron0 = Neuron(RotationAutomorphism(1), ('x',))
neuron1 = Neuron(RotationAutomorphism(2), ('x',))
layer1 = Layer([neuron0, neuron1])

#The second layer has one neuron which is initialized with gAlpha
neuron2 = Neuron(gAlpha, [neuron0, neuron1])
layer2 = Layer([neuron2])

net = NeuralNet([layer0, layer1, layer2])

# The MNIST training set will be used to train this discrete neural net to detect the handwritten digit 0.
# Load some binary images from the modified MNIST training set.
training_pairs = binary_train_for_zero(200)

# We can check out empirical loss with respect to this training set.
# Our loss function will be the Hamming distance.
print("initial loss: "+str(net.empirical_loss(training_pairs, loss_func=lambda x, y: hamming_distance(x[0], y[0]))))
print()

# Use the training set for a list of constant images to use for swapping/blanking endomorphisms.
constant_images = [pair[0]['x'] for pair in training_pairs]

net.train(training_pairs, lambda op: polymorphism_neighbor_func(op, 4, constant_images=constant_images),
          5, lambda x, y: hamming_distance(x[0], y[0]), report_loss=True)



#The current error is because the neighbor function expects gAlpha to take a tuple of images but gAlpha expects two images as separate arguments