"""
Tools for creating random neural nets
"""

import random
from discrete_neural_net import Neuron, Layer, NeuralNet
from operations import Operation, Identity


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

    """

    def __init__(self, basic_ops, previous_layer):
        """

        """

        activation_func = random.choice(basic_ops[random.choice(tuple(basic_ops.keys()))])
        Neuron.__init__(self, activation_func, [random.choice(previous_layer.neurons)
                                                for _ in range(activation_func.arity)])


class RandomLayer(Layer):
    """

    """

    def __init__(self, basic_ops, previous_layer, size):
        """

        """

        Layer.__init__(self, [RandomNeuron(basic_ops, previous_layer) for _ in range(size)])


class RandomNeuralNet(NeuralNet):

    def __init__(self, order, inputs, outputs, depth, breadth, signature):
        """
        Args:
            order (int): The number of elements of the universe.
            inputs (iterable of str): The names of the input neurons.
            outputs (int): The number of output neurons.
            depth (int): The number of layers in the neural net.
            breadth (int): The maximum number of neurons in a layer.
            signature (dict of int: int): Maps an arity to the (nonzero) number of basic operations of that arity.
        """

        basic_ops = {n: [RandomOperation(order, n) for _ in range(signature[n])] for n in signature.keys()}
        nonidentity_basic_ops = basic_ops.copy()
        if 0 in basic_ops.keys():
            basic_ops[1].append(Identity())
        else:
            basic_ops[1] = [Identity()]
        architecture = [Layer(inputs)]
        for _ in range(depth - 2):
            architecture.append(RandomLayer(basic_ops, architecture[-1], random.randint(1, breadth)))
        architecture.append(RandomLayer(nonidentity_basic_ops, architecture[-1], outputs))
        NeuralNet.__init__(self, architecture)
