"""
Arithmetic operations for use as neural net activation functions
"""
from operations import Operation


class ModularAddition(Operation):
    """
    Addition modulo a positive integer.
    """

    def __init__(self, order, cache_values=False):
        """
        Create the addition operation modulo a given positive integer.

        Arguments:
            order (int): The modulus for performing addition.
            cache_values (bool): Whether to memoize the operation.
        """

        # Complain if the order is nonpositive.
        assert order > 0
        Operation.__init__(self, 2, lambda *x: (x[0] + x[1]) % order, cache_values)


class ModularMultiplication(Operation):
    """
    Multiplication modulo a positive integer.
    """

    def __init__(self, order, cache_values=False):
        """
        Create the multiplication operation modulo a given positive integer.

        Arguments:
            order (int): The modulus for performing multiplication.
            cache_values (bool): Whether to memoize the operation.
        """

        # Complain if the order is nonpositive.
        assert order > 0
        Operation.__init__(self, 2, lambda *x: (x[0] * x[1]) % order, cache_values)


class ModularNegation(Operation):
    """
    Negation modulo a positive integer.
    """

    def __init__(self, order, cache_values=False):
        """
        Create the negation operation modulo a given positive integer.

        Arguments:
            order (int): The modulus for performing negation.
            cache_values (bool): Whether to memoize the operation.
        """

        # Complain if the order is nonpositive.
        assert order > 0
        Operation.__init__(self, 1, lambda *x: (-x) % order, cache_values)
