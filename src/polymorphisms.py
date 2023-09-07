"""
Polymorphisms
"""
from relations import Relation
from operations import Projection
from discrete_neural_net import Operation
import random
import numpy


def quarter_turn(rel):
    """
    Rotate a binary relation by a quarter turn counterclockwise.

    Args:
        rel (Relation): The binary relation to be rotated.

    Returns:
        Relation: The same relation rotated by a quarter turn counterclockwise.
    """

    return Relation(((rel.universe_size - tup[1], tup[0]) for tup in rel), rel.universe_size, rel.arity)


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

        def func(x):
            for _ in range(k % 4):
                x = quarter_turn(x)
            return x

        Operation.__init__(self, 1, func=func)


class ReflectionAutomorphism(Operation):
    """
    An automorphism of the Hamming graph obtained by reflecting an image across its vertical axis of symmetry.
    """

    def __init__(self):
        """
        Create a reflection automorphism.
        """

        Operation.__init__(self, 1, lambda rel: Relation(((rel.universe_size - tup[0], tup[1]) for tup in rel),
                                                         rel.universe_size, rel.arity))


class SwappingAutomorphism(Operation):
    """
    An automorphism of the Hamming graph obtained by taking the componentwise xor with a fixed relation.
    """

    def __init__(self, b):
        """
        Create a swapping automorphism for a given relation.

        Argument:
            b (Relation): The fixed relation used for swapping. This must have the same universe and arity as the
                argument passed to the automorphism.
        """

        Operation.__init__(self, 1, lambda a: a ^ b)


class BlankingEndomorphism(Operation):
    """
    An endomorphism of the Hamming graph obtained by taking the intersection with a fixed relation.
    """

    def __init__(self, b):
        """
        Create a blanking endomorphism for a relation.

        Argument:
            b (Relation): The fixed relation used for blanking pixels.
        """

        Operation.__init__(self, 1, lambda a: a & b)


def indicator_polymorphism(tup, a, b):
    """
    Perform an indicator polymorphism where the output is either an empty relation or a relation containing a single
    tuple.

    Args:
        tup (tuple of int): The single tuple in question.
        a (iterable of Relation): A sequence of relations, thought of as inputs to the polymorphism.
        b (iterable of Relation): A sequence of Relations with which dot products are to be taken, thought of as
            constants. This should be the same length as `a`.

    Returns:
        Relation: The relation obtained by applying the indicator polymorphism.
    """

    a = tuple(a)
    universe_size = a[0].universe_size
    if all(rel[0].dot(rel[1]) for rel in zip(a, b)):
        return Relation((tup,), universe_size)
    else:
        return Relation(tuple(), universe_size, len(tup))


class IndicatorPolymorphism(Operation):
    """
    Create a polymorphism of the Hamming graph by taking dot products with fixed relations.
    """

    def __init__(self, tup, b):
        """
        Create an indicator polymorphism where the output is either an empty relation or a relation containing a single
        tuple.

        Args:
            tup (tuple of int): The single tuple in question.
            b (iterable of Relation): A sequence of Relations with which dot products are to be taken, thought of as
            constants. Should contain at least one entry.
        """

        Operation.__init__(self, len(b), lambda *a: indicator_polymorphism(tup, a, b))


def polymorphism_neighbor_func(op, num_of_neighbors, constant_relations, use_dominions=False):
    """
    Find the neighbors of a given polymorphism of the Hamming graph. Currently, this assumes relations are all binary.
    There is also an implicit assumption here that dominion polymorphisms should be binary operations. This could be
    changed as well, but likely is not necessary.

    Arguments:
        op (Operation): A Hamming graph polymorphism operation.
        num_of_neighbors (int): The amount of possible neighbors to generate.
        constant_relations (iterable of Relation): An iterable of relations to use as constants. This is assumed to have
            nonzero length.
        use_dominions (bool): Whether to use dominion polymorphisms. Set the False by default until these are
            working.

    Yields:
        Operation: A neighboring operation to the given one.
    """

    endomorphisms = []
    endomorphisms += [RotationAutomorphism(k) for k in range(4)]
    endomorphisms.append(ReflectionAutomorphism())
    endomorphisms.append('Swapping')
    endomorphisms.append('Blanking')
    constant_relations = tuple(constant_relations)
    universe_size = constant_relations[0].universe_size
    arity = constant_relations[0].arity
    yield op
    for _ in range(num_of_neighbors):
        twist = random.choice((0, 1))
        if twist:
            endomorphisms_to_use = random.choices(endomorphisms, k=op.arity + 1)
            for i in range(len(endomorphisms_to_use)):
                if endomorphisms_to_use[i] == 'Blanking':
                    endomorphisms_to_use[i] = BlankingEndomorphism(random.choice(constant_relations))
                if endomorphisms_to_use[i] == 'Swapping':
                    endomorphisms_to_use[i] = SwappingAutomorphism(random.choice(constant_relations))
            for i in range(len(endomorphisms_to_use)-1):
                endomorphisms_to_use[i] = endomorphisms_to_use[i][Projection(op.arity, i)]
            yield endomorphisms_to_use[-1][op[endomorphisms_to_use[:-1]]]
        else:
            if op.arity == 1:
                random_endomorphism = random.choice(endomorphisms)
                if random_endomorphism == 'Blanking':
                    random_endomorphism = BlankingEndomorphism(random.choice(constant_relations))
                if random_endomorphism == 'Swapping':
                    random_endomorphism = SwappingAutomorphism(random.choice(constant_relations))
                yield random_endomorphism
            if op.arity >= 2:
                if random.randint(0, 1) and use_dominions:
                    pass
                    # This is where the (binary) dominion polymorphisms would be implemented.
                else:
                    # The universe size and relation arity for the indicator polymorphisms is read off from the
                    # `constant_relations`.
                    yield IndicatorPolymorphism(tuple(random.randrange(universe_size) for _ in range(arity)),
                                                random.choices(constant_relations, k=op.arity))


def hamming_loss(x, y):
    """

    Args:
        x:
        y:

    Returns:

    """

    return numpy.average(tuple(len(rel0 ^ rel1) for (rel0, rel1) in zip(x, y)))
