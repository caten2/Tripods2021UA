"""
Discrete neural net
"""

import numpy as np
import random as rand


class Op:

    def __init__(self, arity, func, name='default'):
        self.arity = arity
        self.func = func
        self.name = name

    def eval(self, args):
        return self.func(args)

    def name():
        return self.name


class Neuron:

    def __init__(self, in_edges):
        self.in_edges = in_edges

    def set_op(self, operation):
        self.op = operation

    def fire(self, inputs):
        self.value = self.op(inputs)

    def value(self):
        return self.value


class NeuralNet:

    def __init__(self, architecture, neighborhood_func):
        """
        Parameters
        ----------
        architecture : tuple of tuple of tuple
        
        neighborhood_func : func -> func list
        """
        self.architecture = architecture
        self.nbhd_func = neighborhood_func
        self.layers = tuple(tuple(Neuron) for l in architecture)


