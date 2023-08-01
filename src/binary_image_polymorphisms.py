"""
Polymorphisms for binary images
"""
import random

from discrete_neural_net import Operation
from itertools import product
from dominion import getGAlpha


def quarter_turn(x):
    """
    Rotate a binary image by a quarter turn counterclockwise.

    Args:
        x (list of (list of int)): The binary image to be rotated.

    Returns:
        list of (list of int): The same image rotated by a quarter turn counterclockwise.
    """

    return [[x[i][j] for i in range(len(x))] for j in range(len(x) - 1, -1, -1)]


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
            func = lambda *x: x[0]
        if k % 4 == 1:
            func = lambda *x: quarter_turn(x[0])
        if k % 4 == 2:
            func = lambda *x: quarter_turn(quarter_turn(x[0]))
        if k % 4 == 3:
            func = lambda *x: quarter_turn(quarter_turn(quarter_turn(x[0])))
        Operation.__init__(self, 1, func=func, cache_values=False)


class ReflectionAutomorphism(Operation):
    """
    An automorphism of the Hamming graph obtained by reflecting an image across its vertical axis of symmetry.
    """

    def __init__(self):
        """
        Create a reflection automorphism.
        """

        Operation.__init__(self, 1,
                           lambda *x: [[x[0][i][j] for j in range(len(x[0]) - 1, -1,
                                                                  -1)] for i in range(len(x[0]))],
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
        Operation.__init__(self, 1, lambda *a: [[(a[0][i][j] + b[i][j]) % 2 for j in
                                                 range(size)] for i in range(size)],
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
        Operation.__init__(self, 1, lambda *a: [[a[0][i][j] * b[i][j] for j in range(
            size)] for i in range(size)],
                           cache_values=False)


def dot_product(x, y):
    """
    Take the dot product of two binary images over the finite field of order 2.

    Arguments:
        x (list of (list of int)): The first binary image.
        y (list of (list of int)): The second binary image.

    Returns:
        int: The dot product of the given images.
    """

    size = len(x)
    return sum(x[i][j] * y[i][j] for i, j in product(range(size), repeat=2)) % 2


def indicator_polymorphism(i, j, a, c):
    """
    Perform an indicator polymorphism where the either blank (all zeroes) or the binary image with a
    single black pixel at position (`i`,`j`).

    Args:
        i (int): The row where the single black pixel appears.
        j (int): The column where the single black pixel appears.
        a (iterable of (list of (list of int))): A sequence of binary images with which dot products are to be taken.
        c (iterable of (list of (list of int))): A sequence of binary images with which dot products are to be taken.

    Returns:
        list of (list of int): The binary image obtained by applying the indicator polymorphism.
    """

    size = len(a[0])
    img = [(size * [0])[:] for _ in range(size)]
    if all(dot_product(a[k], c[k]) for k in range(len(a))):
        img[i][j] = 1
    return img


class IndicatorPolymorphism(Operation):
    """
    Create a polymorphism of the Hamming graph by taking dot products with fixed binary images.
    """

    def __init__(self, i, j, c):
        """
        Create an indicator polymorphism where the output image is either blank (all zeroes) or the binary image with a
        single black pixel at position (`i`,`j`).

        Args:
            i (int): The row where the single black pixel appears.
            j (int): The column where the single black pixel appears.
            c (iterable of (list of (list of int))): A sequence of binary images with which dot products are to be
                taken.
        """

        Operation.__init__(self, len(c), lambda *a: indicator_polymorphism(i, j, a, c),
                           cache_values=False)


def polymorphism_neighbor_func(op, num_of_neighbors, constant_images):
    """

    Arguments:
        op (Operation): A binary image polymorphism operation.
        num_of_neighbors (int): The amount of possible neighbors to generate.
        constant_images (list of (list of (list of int))): A list of binary images to use as constants.

    Returns:

    """
    

    endomorphisms = []
    endomorphisms += [RotationAutomorphism(k) for k in range(4)]
    endomorphisms.append(ReflectionAutomorphism())
    endomorphisms.append('Swapping')
    endomorphisms.append('Blanking')
    neighbors = [op]
    for _ in range(num_of_neighbors):
        twist = random.choice((0, 1))
        if twist:
            endomorphisms_to_use = random.choices(endomorphisms, k=op.arity + 1)
            for i in range(len(endomorphisms_to_use)):
                if endomorphisms_to_use[i] == 'Blanking':
                    endomorphisms_to_use[i] = BlankingEndomorphism(random.choice(constant_images))
                if endomorphisms_to_use[i] == 'Swapping':
                    endomorphisms_to_use[i] = SwappingAutomorphism(random.choice(constant_images))
            neighbors.append(Operation(op.arity, lambda *x: endomorphisms_to_use[-1](
                op.func([endomorphisms_to_use[j](x[j],) for j in range(op.arity)]),),
            cache_values=False))
        else:
            if op.arity == 1:
                random_endomorphism = random.choice(endomorphisms)
                if random_endomorphism == 'Blanking':
                    random_endomorphism = BlankingEndomorphism(random.choice(constant_images))
                if random_endomorphism == 'Swapping':
                    random_endomorphism = SwappingAutomorphism(random.choice(constant_images))
                neighbors.append(random_endomorphism)
            if op.arity == 2 and random.randint(0, 1):
                # I would just look at how many trees you have created and hardcode the following line as
                # neighbors.append(getGAlpha(random.randint(0,number_of_trees_you_have-1)))
                # I changed this to 3 for now because currently tree 3 is the only one with the correct size of dominions
                neighbors.append(getGAlpha(3))
            else:
                neighbors.append(IndicatorPolymorphism(random.choice(range(28)), random.choice(range(28)),
                                                       random.choices(constant_images, k=op.arity)))
    return neighbors


def hamming_distance(x, y):
    """
    Compute the Hamming distance between a pair of binary images.

    Args:
        x:
        y:

    Returns:

    """

    return sum(x[i][j] != y[i][j] for i, j in product(range(len(x)), repeat=2))
