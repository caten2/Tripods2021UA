"""
Operations for use as neural net activation functions
"""

import random
from discrete_neural_net import Operation


class RandomOperation(Operation):
    """

    """

    """
    Generate a random operation.
    
    Arguments:
        order (int): Determines the universe of the operation [0, order - 1].
        arity (int): The arity of the random operation.
        
    Returns:
        (Operation): A random operation of arity 'arity' on the universe [0, order - 1].
    """
    def __init__(self, order, arity):
        if arity == 0:
            random_constant = random.randint(0, order - 1)
            Operation.__init__(self, 0, random_constant)
        else:
            Operation.__init__(self, arity, lambda *x: random.randint(0, order - 1))


class ModularAddition(Operation):
    """

    """

    def __init__(self, order, cache_values=False):
        Operation.__init__(self, 2, lambda *x: (x[0] + x[1]) % order, cache_values)


class ModularMultiplication(Operation):
    """

    """

    def __init__(self, order, cache_values=False):
        Operation.__init__(self, 2, lambda *x: (x[0] * x[1]) % order, cache_values)


class ModularNegation(Operation):
    """

    """

    def __init__(self, order, cache_values=False):
        Operation.__init__(self, 1, lambda *x: (-x) % order, cache_values)


class Constant(Operation):
    """
    An operation whose value is `constant` for all inputs. The default arity is 0,
    in which case the correct way to evaluate is as f[()], not f[].
    """

    def __init__(self, constant, order, arity=0, cache_values=False):
        Operation.__init__(self, arity, lambda *x: constant, cache_values)


class Identity(Operation):
    """

    """

    def __init__(self):
        Operation.__init__(self, 1, lambda *x: x[0], cache_values=False)
