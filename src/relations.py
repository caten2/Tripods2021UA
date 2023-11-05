"""
Relations
"""
import random
from itertools import product
from functools import wraps


def comparison(method):
    """
    Check a method for an appropriate comparison by using `self.comparison_check` first.

    Args:
        method (function): The method to which to apply this check.

    Returns:
        function: The given `method` with `self.comparison_check` being called first.
    """

    @wraps(method)
    def checked_method(self, other):
        assert self.comparison_check(other)
        return method(self, other)

    return checked_method



class Relation:
    """
    A finitary relation on a finite set.

    Attributes:
        tuples (frozenset of tuple of int): The tuples belonging to the relation.
        universe_size (int): The number of elements in the universe, which is assumed to consist of an initial section
            of the nonnegative integers.
        arity (int): The length of each tuple in the relation. Can be inferred from `tuples` unless that iterable is
            empty.
    """

    def __init__(self, tuples, universe_size, arity=0):
        """
        Construct a relation from a collection of tuples.

        Arguments:
            tuples (iterable of iterable of int): The tuples belonging to the relation.
            universe_size (int): The number of elements in the universe, which is assumed to consist of an initial
                section of the nonnegative integers.
            arity (int): The length of each tuple in the relation. Can be inferred from `tuples` unless that iterable is
                empty.
        """

        # Create a tuple of tuples of integers from the given iterable `tuples`.
        tuples = tuple(tuple(entry) for entry in tuples)
        # If `tuples` is empty then we have an empty relation and cannot infer its arity from its members.
        # If no value is provided for the arity, it defaults to 0.
        if len(tuples):
            # We assume that all entries in `tuples` have the same length. Take one of them to get the arity of the
            # relation.
            self._arity = len(tuples[0])
        else:
            self._arity = arity
        # Cast `tuples` to a frozenset and store it as the `tuples` attribute of the relation.
        self._tuples = frozenset(tuples)
        # Store the size of the universe.
        self._universe_size = universe_size

    @property
    def tuples(self):
        return self._tuples

    @property
    def universe_size(self):
        return self._universe_size

    @property
    def arity(self):
        return self._arity

    def __len__(self):
        """
        Give the number of tuples in the relation.

        Returns:
            int: The number of tuples in `self.tuples`.
        """

        return len(self.tuples)

    def __str__(self):
        """
        Display basic information about the relation.

        Returns:
            str: Information about the universe, arity, and size of the relation.
        """

        # When the universe size is large we use ellipsis rather than write out the whole universe.
        if self.universe_size > 10:
            universe = '{0,...,' + str(self.universe_size - 1) + '}'
        else:
            universe = '{' + ','.join(map(str, range(self.universe_size))) + '}'
        # Check whether 'tuple' needs to be pluralized.
        if len(self) == 1:
            plural = ''
        else:
            plural = 's'
        return 'A relation on {} of arity {} containing {} tuple{}'.format(universe, self.arity, len(self), plural)

    def __contains__(self, tup):
        """
        Check whether a tuple belongs to a relation.

        Argument:
            tup (tuple of int): The tuple we are checking.

        Returns:
            bool: True when `tup` belongs to `self.tuples`, False otherwise.
        """

        return tup in self.tuples

    def __iter__(self):
        """
        Produce an iterator for the tuples in the relation.

        Returns:
            frozenset: The set of tuples in the relation.
        """

        return iter(self.tuples)

    def __bool__(self):
        """
        Cast a relation to a boolean value.

        Returns:
            bool: True when self.tuples is nonempty, False otherwise.
        """

        return bool(self.tuples)

    def show(self, special_binary_display=None):
        """
        Display the members of `self.tuples`.

        Arguments:
            special_binary_display (str): Show a binary relation through some other method than just printing pairs.
                The default is None, which means pairs will be printed as usual. This can be set to 'binary_pixels' in
                order to print out the binary image corresponding to the relation or 'sparse' to display the presence of
                a pair as an X and the absence of a pair as a space.
        """

        if special_binary_display:
            if special_binary_display == 'binary_pixels':
                for row in range(self.universe_size):
                    line = ''
                    for column in range(self.universe_size):
                        if (row, column) in self:
                            line += '1'
                        else:
                            line += '0'
                    print(line)
            if special_binary_display == 'sparse':
                for row in range(self.universe_size):
                    line = ''
                    for column in range(self.universe_size):
                        if (row, column) in self:
                            line += 'X'
                        else:
                            line += ' '
                    print(line)
            if special_binary_display == 'latex_matrix':
                output = '\\begin{bmatrix}'
                for row in range(self.universe_size):
                    output += '&'.join(map(str, (int((row, column) in self) for column in range(self.universe_size))))
                    if row != self.universe_size-1:
                        output += '\\\\'
                output += '\\end{bmatrix}'
                print(output)
        else:
            for tup in self:
                print(tup)

    def comparison_check(self, other):
        """
        Determine whether another `Relation` object is of the correct type to be comparable with the relation in
        question.

        Argument:
            other (Relation): The other relation to which to compare this relation.

        Returns:
            bool: True when the two relations have the same universe and arity, False otherwise.
        """

        return self.universe_size == other.universe_size and self.arity == other.arity

    def __hash__(self):
        """
        Find the hash value for the `Relation` object.

        Returns:
            int: The hash value of the `Relation` object.
        """

        return hash((self.tuples, self.universe_size, self.arity))

    @comparison
    def __eq__(self, other):
        """
        Check whether the relation is equal to another relation.

        Argument:
            other (Relation): The other relation to which to compare this relation.

        Returns:
            bool: True when self.tuples is equal to other.tuples and False otherwise.
        """

        return self.tuples == other.tuples

    @comparison
    def __lt__(self, other):
        """
        Check whether the relation is properly contained in another relation.

        Argument:
            other (Relation): The other relation to which to compare this relation.

        Returns:
            bool: True when self.tuples is a proper subset of other.tuples and False otherwise.
        """

        return self.tuples < other.tuples

    @comparison
    def __le__(self, other):
        """
        Check whether the relation is contained in another relation.

        Argument:
            other (Relation): The other relation to which to compare this relation.

        Returns:
            bool: True when self.tuples is a subset of other.tuples and False otherwise.
        """

        return self.tuples <= other.tuples

    @comparison
    def __gt__(self, other):
        """
        Check whether the relation properly contains in another relation.

        Argument:
            other (Relation): The other relation to which to compare this relation.

        Returns:
            bool: True when self.tuples is a proper superset of other.tuples and False otherwise.
        """

        return self.tuples > other.tuples

    @comparison
    def __ge__(self, other):
        """
        Check whether the relation contains in another relation.

        Argument:
            other (Relation): The other relation to which to compare this relation.

        Returns:
            bool: True when self.tuples is a superset of other.tuples and False otherwise.
        """

        return self.tuples >= other.tuples

    def __invert__(self):
        """
        Create the complement of a relation. That is, a tuple in the appropriate Cartesian power of the universe will
        belong to the complement if and only if it does not belong to the given relation.

        Returns:
            Relation: The relation which is dual to the given relation in the above sense.
        """

        return Relation((tup for tup in product(range(self.universe_size), repeat=self.arity) if tup not in self),
                        self.universe_size, self.arity)

    @comparison
    def __sub__(self, other):
        """
        Take the difference of two relations. This is the same as the set difference of their sets of tuples.

        Argument:
            other (Relation): The other relation to remove from this relation.

        Returns:
            Relation: The relation with the same universe and arity as the inputs which is their set difference.
        """

        return Relation(self.tuples.difference(other.tuples), self.universe_size, self.arity)

    @comparison
    def __and__(self, other):
        """
        Take the intersection of two relations. This is the same as bitwise multiplication.

        Argument:
            other (Relation): The other relation to intersect with this relation.

        Returns:
            Relation: The relation with the same universe and arity as the inputs which is their intersection.
        """

        return Relation(self.tuples.intersection(other.tuples), self.universe_size, self.arity)

    @comparison
    def __or__(self, other):
        """
        Take the union of two relations. This is the same as bitwise disjunction.

        Argument:
            other (Relation): The other relation to union with this relation.

        Returns:
            Relation: The relation with the same universe and arity as the inputs which is their union.
        """

        return Relation(self.tuples.union(other.tuples), self.universe_size, self.arity)

    @comparison
    def __xor__(self, other):
        """
        Take the symmetric difference of two relations. This is the same as bitwise addition.

        Argument:
            other (Relation): The other relation to which to add this relation.

        Returns:
            Relation: The relation with the same universe and arity as the inputs which is their symmetric difference.
        """

        return Relation(self.tuples.symmetric_difference(other.tuples), self.universe_size, self.arity)

    @comparison
    def __isub__(self, other):
        """
        Take the set difference of two relations with augmented assignment.

        Argument:
            other (Relation): The other relation to remove from this relation.

        Returns:
            Relation: The relation with the same universe and arity as the inputs which is their set difference.
        """

        return self - other

    @comparison
    def __iand__(self, other):
        """
        Take the set intersection of two relations with augmented assignment.

        Argument:
            other (Relation): The other relation to remove from this relation.

        Returns:
            Relation: The relation with the same universe and arity as the inputs which is their intersection.
        """

        return self & other

    @comparison
    def __ior__(self, other):
        """
        Take the set union of two relations with augmented assignment.

        Argument:
            other (Relation): The other relation to union with this relation.

        Returns:
            Relation: The relation with the same universe and arity as the inputs which is their union.
        """

        return self | other

    @comparison
    def __ixor__(self, other):
        """
        Take the symmetric difference of two relations with augmented assignment.

        Argument:
            other (Relation): The other relation to which to add this relation.

        Returns:
            Relation: The relation with the same universe and arity as the inputs which is their symmetric difference.
        """

        return self ^ other

    @comparison
    def dot(self, other):
        """
        Take the dot product of two relations modulo 2. This is the same as computing the size of the intersection of
        the two relations modulo 2.

        Argument:
            other (Relation): The other relation with to take the dot product.

        Returns:
            int: Either 0 or 1, depending on the parity of the number of tuples in `self` and `other`.
        """

        return len(self & other) % 2


def random_atomic_relations(universe_size, arity):
    """
    Create randomly-chosen atomic relations.

    Args:
        universe_size:
        arity:

    Yields:

    """

    while True:
        tup = tuple(random.randrange(universe_size) for _ in range(arity))
        yield Relation((tup, ), universe_size)

