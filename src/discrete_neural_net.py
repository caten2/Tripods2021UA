"""
Discrete neural net
"""

import random
import numpy


class Operation:
    """
    A finitary operation on a set.

    Attributes:
        arity (int): The number of arguments the operation takes. This quantity should be at least 0. A 0-ary
            Operation takes empty tuples as arguments. See the method __getitem__ below for more information on this.
        func (function or constant): The function which is used to compute the output
        value of the Operation when applied to some
            inputs.
        cache_values (bool): Whether to store already-computed values of the Operation in memory.
        values (dict): If `cache_values` is True then this attribute will keep track of which input-output pairs have
            already been computed for this Operation so that they may be reused. This can be replaced by another object
            that can be indexed.
    """

    def __init__(self, arity, func, cache_values=True):
        """
        Create a finitary operation on a set.

        Arguments:
            arity (int): The number of arguments the operation takes. This quantity should be at least 0. A 0-ary
                Operation takes empty tuples as arguments. See the method __getitem__ below for more information on
                this.
            func (function): The function which is used to compute the output value of the Operation when applied to
                some inputs. If the arity is 0, pass a constant, not a function, here.
            cache_values (bool): Whether to store already-computed values of the Operation in memory.
        """

        self.arity = arity
        self.func = func
        self.cache_values = cache_values
        if self.cache_values:
            self.values = {}

    def __call__(self, *index):
        """
        Compute the value of the Operation on given inputs.

        Argument:
            index (tuple): The tuple of inputs to plug in to the Operation.
        """
        if self.arity == 0:
            return self.func
        if self.cache_values:
            if index not in self.values.keys():
                self.values[index] = self.func(*index)
            return self.values[index]
        return self.func(*index)


class Neuron:
    """
    A neuron in a neural net.

    Attributes:
        activation_func (Operation): The activation function of the neuron.
        inputs (list of Neuron): The neurons which act as inputs to the neuron in question.
    """

    def __init__(self, activation_func, inputs):
        """
        Construct a neuron for use in a neural net.

        Arguments:
            activation_func (Operation): The activation function of the neuron.
            inputs ((tuple of str) or (list of Neuron)): The neurons which act as inputs to the neuron in question.
        """

        self.activation_func = activation_func
        self.inputs = inputs


class Layer:
    """
    A layer in a neural net.

    Attribute:
        neurons ((tuple of str) or (list of Neuron)): If `neurons` is a tuple of str then we take the corresponding
                Layer object to be an input layer for a neural net, with the entries of `neurons` being distinct
                variable names for the arguments to the neural net.
    """

    def __init__(self, neurons):
        """
        Construct a layer with a given collection of neurons.

        Argument:
            neurons ((tuple of str) or (list of Neuron)): If `neurons` is a tuple of str then we take the corresponding
                Layer object to be an input layer for a neural net, with the entries of `neurons` being distinct
                variable names for the arguments to the neural net.
        """

        self.neurons = neurons


def zero_one_loss(x, y):
    """
    Compute the 0-1 loss for a given pair of tuples.
    The input tuples should have the same length.

    Arguments:
        x (tuple): A tuple of outputs from feeding forward through a neural net.
        y (tuple): A tuple of target outputs from a training set.

    Returns:
        int: Either 0 (the tuples agree) or 1 (the tuples do not agree).
    """

    return 1-(x == y)


class NeuralNet:
    """
    A (discrete) neural net.

    Attribute:
        architecture (list of Layer): The layers of the neural net, starting with the input layer,
            whose neurons should be a list of distinct variable names. Later layers should consist
            of Neurons carrying activation functions.
    """

    def __init__(self, architecture):
        """
        Construct a neural net with a given architecture.

        Argument:
            architecture (list of Layer): The layers of the neural net, starting with the input layer,
                whose neurons should be a list of distinct variable names. Later layers should consist
                of Neurons carrying activation functions.
        """

        self.architecture = architecture

    def feed_forward(self, x):
        """
        Feed the values `x` forward through the neural net.

        Argument:
            x (dict of str: object): An assignment of variable names to values.

        Returns:
            tuple: The current values of each of the output layer neurons after feeding forward.
        """

        current_vals = x
        for layer in self.architecture[1:]:
            for neuron in layer.neurons:
                index = tuple(current_vals[input_neuron] for input_neuron in neuron.inputs)
                current_vals[neuron] = neuron.activation_func(*index)
        return tuple(current_vals[neuron] for neuron in self.architecture[-1].neurons)

    def empirical_loss(self, training_pairs, loss_func=zero_one_loss):
        """
        Calculate the current empirical loss of the neural net with respect to the training pairs and loss function.

        Argument:
            training_pairs (iterable): Training pairs (x,y) where x is a dictionary of inputs and y is a tuple of
                outputs.
            loss_func (function): The loss function to use for training. The default is the 0-1 loss.

        Returns:
            numpy.float64: The empirical loss. This is a float between 0 and 1, with 0 meaning our model is perfect on
                the training set and 1 being complete failure.
        """

        # Create a list of loss function values for each pair in our training set, then average them.
        return numpy.average([loss_func(self.feed_forward(x), y) for (x, y) in training_pairs])

    def training_step(self, training_pairs, neighbor_func, loss_func=zero_one_loss):
        """
        Perform one step of training the neural net using the given training pairs, neighbor function,
        and loss function. At each step a random non-input neuron is explored. The neighbor function tells us which
        other activation functions we should try in place of the one already present at that neuron. We use the loss
        function and the training pairs to determine which of these alternative activation functions we should use at
        the given neuron instead.

        Arguments:
            training_pairs (iterable): Training pairs (x,y) where x is a dictionary of inputs and y is a tuple of
                outputs.
            neighbor_func (function): A function which takes an Operation as input and returns an iterable of
                Operations as output.
            loss_func (function): The loss function to use for training. The default is the 0-1 loss.
        """

        # Select a random non-input layer from the neural net.
        layer = random.choice(self.architecture[1:])
        # Choose a random neuron from that layer.
        neuron = random.choice(layer.neurons)
        # Store a list of all the adjacent operations given by the supplied neighbor function.
        ops = neighbor_func(neuron.activation_func)
        # Also keep a list of the empirical loss associated with each of the operations in `ops`.
        emp_loss = []
        # Try each of the operations in `ops`.
        for neighbor_op in ops:
            # Change the activation function of `neuron` to the current candidate under consideration.
            neuron.activation_func = neighbor_op
            # Add the corresponding empirical loss (the average of the loss values) to the list of empirical losses.
            emp_loss.append(self.empirical_loss(training_pairs, loss_func))
        # Conclude the training step by changing the activation function of `neuron` to the candidate activation
        # function which results in the lowest empirical loss.
        neuron.activation_func = ops[emp_loss.index(min(emp_loss))]

    def train(self, training_pairs, neighbor_func, iterations, loss_func=zero_one_loss, report_loss=False):
        """
        Train the neural net by performing the training step repeatedly.

        Arguments:
            training_pairs (iterable): Training pairs (x,y) where x is a dictionary of inputs and y is a tuple of
                outputs.
            neighbor_func (function): A function which takes an Operation as input and returns an iterable of
                Operations as output.
            loss_func (function): The loss function to use for training. The default is the 0-1 loss.
            iterations (int): The number of training steps to perform.
            report_loss (bool): Whether to print the final empirical loss after the training has concluded
        """

        for _ in range(iterations):
            self.training_step(training_pairs, neighbor_func, loss_func)
        if report_loss:
            print(self.empirical_loss(training_pairs, loss_func))
