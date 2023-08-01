from pathlib import Path

path = str(Path(__file__).parent.absolute() / "src")
import sys
sys.path.insert(0, path)

from discrete_neural_net import Operation
from itertools import product
import random

class Relation:

    def __init__(self, graphOrList, isGraph, n, arity=2):
        """
        Parameters: 
            graphOrList (list of (list of int)) or (list of (tuple of int)): the input of the relation.
                This can either be inputted as a binary image (list of list of int) or as the subset
                of A^{arity} in the way that a relation is mathematically defined
                (list of tuple of int, where the length of each tuple corresponds to the arity).
            isGraph (boolean): if the input is a list of list of integers, this is True. We will then 
                change the format of the binary image to a list of tuples of integers, in the way that
                a relatio is typically defined. If the input is a list of tuples of integers, isGraph is
                False, as we do not need to change the input's format.
            n (int): the length of the binary image to be represented as a relation.
            arity (int = 2): the number of elements within one tuple of the relation. This should be at
                least 0, which would be the empty relation.

        Attributes: 
            rList (list of (tuple of(int))): the relation represented as a list of tuples of integers.
            n (int): the length of the binary image to be represented as a relation.
        """

        if isGraph:
            self.rList = [(x, y) for x in range(len(graphOrList[0])) for y in range(len(graphOrList[0])) if graphOrList[x][y] == 1]
        else:
            self.rList = graphOrList
        
        self.n = n



def quarter_turn(x):
    """
    Rotate an image by a quarter turn counterclockwise.

    Args:
        x (Relation): The binary image to be rotated, represented by a relation.

    Returns:
        Relation: The same image rotated by a quarter turn counterclockwise, represented by
            a relation.
    """

    return [(x.n - (x.rList[k][1]+1), x.rList[k][0]) for k in range(len(x.rList))]



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
            func = lambda x: x[0] # apply new operaiton
        if k % 4 == 1:
            func = lambda x: quarter_turn(x[0])
        if k % 4 == 2:
            func = lambda x: quarter_turn(quarter_turn(x[0]))
        if k % 4 == 3:
            func = lambda x: quarter_turn(quarter_turn(quarter_turn(x[0])))
        Operation.__init__(self, 1, func=func, cache_values=False)



def dot_product(x, y):
    """
    Take the dot product of two binary images over the finite field of order 2.

    Arguments:
        x (Relation): The first binary image.
        y (Relation): The second binary image.
    Returns:
        int: The dot product of the given images.
    """

    return len([(i, j) for (i, j) in x.rList if (i, j) in y.rList]) % 2



def indicator_polymorphism(i, j, a, c):
    """
    Perform an indicator polymorphism where the either blank (all zeroes) or the binary image with a
    single black pixel at position (`i`,`j`).

    Args:
        i (int): The row where the single black pixel appears.
        j (int): The column where the single black pixel appears.
        a (iterable of Relation): A sequence of binary images with which dot products are to be taken.
        c (iterable of Relation): A sequence of binary images with which dot products are to be taken.

    Returns:
        Relation: The binary image obtained by applying the indicator polymorphism.
    """

    if all(dot_product(a[k], c[k]) for k in range(len(a))):
        return [(i, j)]
    return []



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
            c (iterable of Relation): A sequence of binary images with which dot products are to be
                taken.
        """

        Operation.__init__(self, len(c), lambda a: indicator_polymorphism(i, j, a, c), cache_values=False)



def hamming_distance(x, y):
    """
    Compute the Hamming distance between a pair of binary images.

    Args:
        x (Relation): the first binary image
        y (Relation): the second binary image

    Returns:
        int: the Hamming distance between x and y, ie the number of pixels at which they differ.

    """
    return len([(i, j) for (i, j) in x.rList if (i, j) not in y.rList]) + len([(i, j) for (i, j) in y.rList if (i, j) not in x.rList])


def reflectionVertical(x):
    """
    Reflect an image across its vertical axis of symmetry.

    Args:
        x (Relation): The binary image to be reflected, represented by a relation.

    Returns:
        Relation: The same image reflected across its vertical axis of symmetry, represented by
            a relation.
    """
    return [(i, int((x.n - j -1)%x.n)) for (i, j) in x.rList]



class ReflectionAutomorphism(Operation):
    """
    An automorphism of the Hamming graph obtained by reflecting an image across its vertical axis of symmetry.
    """

    def __init__(self):
        """
        Create a reflection automorphism.

        """
        

        Operation.__init__(self, 1, lambda x: reflectionVertical(x[0]), cache_values=False)


def swapping(x, y):
    """
    Taking the component-wise sum of two binary images.

    Args:
        x (Relation): The first binary image to be added
        y (Relation): The second binary image to be added

    Returns:
        Relation: The image with entries of componentwise sums of x and y
    
    """

    return [(i, j) for (i, j) in x.rList if (i, j) not in y.rList] + [(i, j) for (i, j) in y.rList if (i, j) not in x.rList]




class SwappingAutomorphism(Operation): 
    """
    An automorphism of the Hamming graph obtained by taking the componentwise sum with a fixed binary image.
    """

    def __init__(self, b):
        """
        Create a swapping automorphism for a given image.

        Argument:
            b (Relation): The fixed binary image used for swapping pixels, represented as a relation
        """

        size = len(b)
        Operation.__init__(self, 1, lambda a: swapping(a, b),
                           cache_values=False)



def blanking(x, y):
    """
    Taking the Hadamard product (component-wise multiplication) of two binary images.

    Args:
        x (Relation): The first binary image in the Hadamard product
        y (Relation): The second binary image in the Hadamard product

    Returns:
        Relation: The image of Hadamard product of x and y
    
    """

    return [(i, j) for (i, j) in x.rList if (i, j) in y.rList]




class BlankingEndomorphism(Operation):
    """
    An endomorphism of the Hamming graph obtained by taking the Hadamard product with a fixed binary image.
    """

    def __init__(self, b):
        """
        Create a blanking endomorphism for a given image.

        Argument:
            b (Relation): The fixed binary image used for swapping pixels, represented by relation.
        """

        size = len(b)
        Operation.__init__(self, 1, lambda a: swapping(a, b), cache_values=False)




def polymorphism_neighbor_func(op, num_of_neighbors, constant_images):
    """

    Arguments:
        op (Operation): A binary image polymorphism operation.
        num_of_neighbors (int): The amount of possible neighbors to generate.
        constant_images (list of Relation): A list of binary images (as relations) to use as constants.

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
            neighbors.append(Operation(op.arity, lambda x: endomorphisms_to_use[-1][
                op.func([endomorphisms_to_use[j][x[j],] for j in range(op.arity)]),], cache_values=False))
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






graph = [
    [0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0,],
    [1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0,]
]

graph2 = [
    [0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1,],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1,]
]

a = Relation(graph, True, 6)
a2 = Relation(graph2, True, 6)

print(quarter_turn(a))
print(dot_product(a, a2))
print(indicator_polymorphism(1, 1, [a, a2], [a, a2]))
print(hamming_distance(a, a2))
print(reflectionVertical(a))
print(swapping(a, a2))
print(blanking(a, a2))

