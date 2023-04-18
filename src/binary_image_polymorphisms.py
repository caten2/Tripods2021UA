"""
Polymorphisms for binary images
"""

from discrete_neural_net import Operation
from itertools import product


def quarter_turn(x):
    """
    Rotate a binary image by a quarter turn counterclockwise.

    Args:
        x (list of (list of int)): The binary image to be rotated.

    Returns:
        list of (list of int): The same image rotated by a quarter turn counterclockwise.
    """

    return [[x[i][j] for i in range(len(x))] for j in range(len(x))[::-1]]


class RotationAutomorphism(Operation):
    """
    An automorphism of the Hamming graph obtained by rotating an image.
    """

    def __init__(self, k=1):
        """
        Create a rotation automorphism.

        Argument:
            k (int): The number of quarter turns by which to rotate the image counterclockwise.
        """

        if k % 4 == 0:
            func = lambda x: x
        if k % 4 == 1:
            func = quarter_turn
        if k % 4 == 2:
            func = lambda x: quarter_turn(quarter_turn(x))
        if k % 4 == 3:
            func = lambda x: quarter_turn(quarter_turn(quarter_turn(x)))
        Operation.__init__(self, 1, func=func, cache_values=False)


class ReflectionAutomorphism(Operation):
    """
    An automorphism of the Hamming graph obtained by reflecting an image across its vertical axis of symmetry.
    """

    def __init__(self):
        """
        Create a reflection automorphism.
        """

        Operation.__init__(self, 1, lambda x: [[x[i][j] for j in range(len(x))[::-1]] for i in range(len(x))],
                           cache_values=False)


class SwappingAutomorphism(Operation):
    """
    An automorphism of the Hamming graph obtained by taking the componentwise sum with a fixed binary image.
    """

    def __init__(self, b):
        """
        Create a swapping automorphism for a given image.

        Argument:
            b (list of (list of int)): The fixed binary image used for swapping pixels.
        """

        size = len(b)
        Operation.__init__(self, 1, lambda a: [[(a[i][j] + b[i][j]) % 2 for j in range(size)] for i in range(size)],
                           cache_values=False)


class BlankingEndomorphism(Operation):
    """
    An endomorphism of the Hamming graph obtained by taking the Hadamard product with a fixed binary image.
    """

    def __init__(self, b):
        """
        Create a blanking endomorphism for a given image.

        Argument:
            b (list of (list of int)): The fixed binary image used for swapping pixels.
        """

        size = len(b)
        Operation.__init__(self, 1, lambda a: [[a[i][j] * b[i][j] for j in range(size)] for i in range(size)],
                           cache_values=False)


def dot_product(x,y):
    """
    Take the dot product of two binary images over the finite field of order 2.

    Arguments:
        x (list of (list of int)): The first binary image.
        y (list of (list of int)): The second binary image.

    Returns:
        int: The dot product of the given images.
    """

    size = len(x)
    return sum(x[i][j] for i,j in product(range(size), repeat=2)) % 2

