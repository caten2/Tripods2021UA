"""
Tools for creating random neural nets
"""

import random
from neural_net import Neuron, Layer, NeuralNet
from operations import Operation


class RandomOperation(Operation):
    """
    A random operation. The values of the operation on its arguments are chosen randomly and lazily, but they are
    memoized so that the operation is well-defined.
    """

    def __init__(self, order, arity):
        """
        Create a random operation of a given arity and order.

        Arguments:
            order (int): The size of the universe.
            arity (int): The arity of the operation.
        """

        if arity == 0:
            # For a nullary operation, we choose a random member of the universe to be the corresponding constant.
            random_constant = random.randint(0, order - 1)
            Operation.__init__(self, 0, random_constant)
        else:
            Operation.__init__(self, arity, lambda *x: random.randint(0, order - 1))


class RandomNeuron(Neuron):
    """
    A random neuron. The activation function of the neuron will be chosen from a provided list of possibilities
    and the inputs of the neuron will be chosen from a provided previous layer.
    """

    def __init__(self, basic_ops, previous_layer):
        """
        Create a random neuron.

        Arguments:
            basic_ops (dict of int: iterable): The keys of this dictionary are arities and the values are iterables of
                `Operation`s of that arity.
            previous_layer (Layer): The preceding layer from which inputs are taken.
        """

        activation_func = random.choice(basic_ops[random.choice(tuple(basic_ops.keys()))])
        Neuron.__init__(self, activation_func, [random.choice(previous_layer.neurons)
                                                for _ in range(activation_func.arity)])


class RandomLayer(Layer):
    """
    A random layer consisting of random neurons.
    """

    def __init__(self, basic_ops, previous_layer, size):
        """
        Create a random layer. This takes the same dictionary of basic operations as the `RandomNeuron` constructor, as
        well as a previous layer and a desired number of nodes.

        Arguments:
            basic_ops (dict of int: iterable): The keys of this dictionary are arities and the values are iterables of
                `Operation`s of that arity.
            previous_layer (Layer): The preceding layer from which inputs are taken.
            size (int): The number of nodes to include in the random layer.
        """

        Layer.__init__(self, [RandomNeuron(basic_ops, previous_layer) for _ in range(size)])


class RandomNeuralNet(NeuralNet):
    """
    A neural net whose architecture and activation functions are chosen randomly.
    """

    def __init__(self, basic_ops, inputs, outputs, depth, breadth):
        """
        Create a random neurol net with a given collection of basic activation functions. The breadth and depth of the
        net should be specified, as well as the number of inputs/outputs, but otherwise the architecture is chosen
        randomly.

        Arguments:
            basic_ops (dict of int: iterable): The keys of this dictionary are arities and the values are iterables of
                `Operation`s of that arity.
            inputs (iterable of str): The names of the input neurons.
            outputs (int): The number of output neurons.
            depth (int): The number of layers in the neural net.
            breadth (int): The maximum number of neurons in a layer.
        """

        architecture = [Layer(inputs)]
        for _ in range(depth - 2):
            architecture.append(RandomLayer(basic_ops, architecture[-1], random.randint(1, breadth)))
        architecture.append(RandomLayer(basic_ops, architecture[-1], outputs))
        NeuralNet.__init__(self, architecture)
