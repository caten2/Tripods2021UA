"""
Polymorphisms
"""
from relations import Relation
from operations import Operation


def quarter_turn(rel):
    """
    Rotate a binary relation by a quarter turn counterclockwise.

    Args:
        rel (Relation): The binary relation to be rotated.

    Returns:
        Relation: The same relation rotated by a quarter turn counterclockwise.
    """

    return Relation(((rel.universe_size - tup[1], tup[0]) for tup in rel), rel.universe_size)


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
        Operation.__init__(self, 1, func=func, cache_values=True)

