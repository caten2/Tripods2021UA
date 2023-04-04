"""
Operations for use as neural net activation functions
"""

import random as rand
from discrete_neural_net import Operation


class RandomOperation(Operation):
    """

    """

    def __init__(self, order, arity):
        Operation.__init__(self, order, arity, lambda x: rand.randint(0, self.order -1))


class ModularAddition(Operation):
    """

    """

    def __init__(self, order, cache_values=False):
        Operation.__init__(self, order, 2, lambda x: (x[0] + x[1]) % order, cache_values)


class ModularMultiplication(Operation):
    """

    """

    def __init__(self, order, cache_values=False):
        Operation.__init__(self, order, 2, lambda x: (x[0] * x[1]) % order, cache_values)


class ModularNegation(Operation):
    """

    """

    def __init__(self, order, cache_values=False):
        Operation.__init__(self, order, 1, lambda x: (-x) % order, cache_values)


class Constant(Operation):
    """
    An operation whose value is `constant` for all inputs. The default arity is 0,
    in which case the correct way to evaluate is as f[()], not f[].
    """

    def __init__(self, constant, order, arity=0, cache_values=False):
        Operation.__init__(self, order, arity, lambda x: constant, cache_values)


class Identity(Operation):
    """

    """

    def __init__(self ,order):
        Operation.__init__(self, order, 1, lambda x: x[0], cache_values=False)
