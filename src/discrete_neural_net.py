"""
Discrete neural net
"""

import numpy as np
import random
# from graphviz import Graph


class Operation:
    """

    """

    def __init__(self, order, arity, func, cache_values=True):
        """

        """

        self.order = order
        self.arity = arity
        self.func = func
        self.cache_values = cache_values
        if self.cache_values:
            self.values = -np.ones(arity * [order])

    def __getitem__(self, index):
        """

        """

        if self.cache_values and self.arity == 0:
            if self.values == -1:
                self.values = self.func(index)
        if self.cache_values and self.arity > 0:
            if self.values[index] == -1:
                self.values[index] = self.func(index)
            return int(self.values[index])
        return self.func(index)


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
            inputs (list of Neuron): The neurons which act as inputs to the neuron in question.
        """

        self.activation_func = activation_func
        self.inputs = inputs


class Layer:
    """

    """

    def __init__(self, neurons):
        self.neurons = neurons


class NeuralNet:
    """
    A (discrete) neural net.

    Attributes:
        architecture (list of Layer): The layers of the neural net, starting with the input layer,
            whose neurons should be a list of distinct variable names. Later layers should consist
            of Neurons carrying activation functions.
    """

    def __init__(self, architecture):
        """

        """

        self.architecture = architecture

    def feed_forward(self, x):
        """
        Feed the values `x` forward through the neural net.

        Arguments:
            x (dict of str: int): An assignment of variable names to values.

        Returns:
            tuple: The current values of each of the output layer neurons after feeding forward.
        """

        current_vals = x
        for layer in self.architecture[1:]:
            for neuron in layer.neurons:
                index = tuple(current_vals[input_neuron] for input_neuron in neuron.inputs)
                current_vals[neuron] = neuron.activation_func[index]
        return tuple(current_vals[neuron] for neuron in self.architecture[-1].neurons)

    def train(self, training_pairs, neighbor_func, loss_func):
        """

        Arguments:
            training_pairs (iterable): Training pairs (x,y) where x is a dictionary of inputs and y is a tuple of
                outputs.
            neighbor_func (function): A function which takes an Operation as input and returns an iterable of
                Operations as output.
            loss_func (function): The loss function to use for training.
        """

        layer = random.choice(self.architecture[1:])
        neuron = random.choice(layer.neurons)
        operations = []
        emp_loss = []
        for neighbor_op in neighbor_func(neuron):
            vals = []
            for (x,y) in training_pairs:
                neuron.activation_func = neighbor_op
                vals.append(loss_func(self.feed_forward(x),y))
            operations.append(neighbor_op)
            emp_loss.append(np.average(vals))
        neuron.activation_func = operations[emp_loss.index(min(emp_loss))]

    # def to_graphviz(self, caption: str, show_vals: bool = False) -> Graph:
    #     """
    #     Parameters
    #     ----------
    #     caption : caption for the graph.
    #     show_vals : whether to display the current values of neurons.
    #     """
    #
    #     def helper(i, g):
    #         id = str(i + 1)
    #         prev_layer_neuron_count = self.arity if i == 0 else len(self.layer_at(i - 1))
    #         with g.subgraph(name="cluster_" + id) as c:
    #             c.node_attr['style'] = 'filled'
    #             c.node_attr['color'] = 'lightblue'
    #             c.attr(color='none', label='L' + id)
    #             for j, neuron in enumerate(self.layer_at(i)):
    #                 node_label = neuron.op_as_str()
    #                 if show_vals: node_label += '\\n' + str(neuron.value)
    #                 c.node(id + '_' + str(j), label=node_label, shape='square', style='rounded')
    #                 id_neuron = id + '_' + str(j)
    #                 for k in range(prev_layer_neuron_count):
    #                     if k in neuron.in_edges:
    #                         c.edge(str(i) + '_' + str(k), id_neuron)
    #                     else:
    #                         c.edge(str(i) + '_' + str(k), id_neuron, style='invis')
    #
    #     # >>>>>>>>>> end of helper func <<<<<<<<<<#
    #     g = Graph()
    #     g.attr('graph', constraint='false', clusterrank='local')
    #     g.node_attr['fontname'] = 'helvetica'
    #     g.attr('graph', splines='false', rankdir='LR', ranksep='1.5', label=caption)
    #     with g.subgraph(name='cluster_0') as c:  # input cluster
    #         c.attr(color='none', label='input')
    #         for j, var in enumerate(self.input_layer):
    #             c.node('0_' + str(j), label=str(var), shape='circle')
    #     for i in range(len(self.layers) - 1):
    #         helper(i, g)
    #     i = len(self.layers) - 1
    #     helper(i, g)
    #     return g
