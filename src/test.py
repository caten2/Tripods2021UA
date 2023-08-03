from pathlib import Path

path = str(Path(__file__).parent.absolute() / "src")
import sys
sys.path.insert(0, path)

from discrete_neural_net import Operation
from itertools import product
import random
import numpy as np


# Note that we have added a hyperoctohedral polymorphism, but have kept in the equivalent
# 2-cube polymorphisms (rotation, reflection). If the hyperoctohedral polymorphism works,
# we can delete the rotation and reflection polymorphisms, but we didn't want to delete 
# something that works.

# Also, if ^ works, we would need to put it into the neighbor function (and remove rotation
# and reflection endomorphisms from the function)

# (づ ◕‿◕ )づ
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
            n (int): the length of the n-ary image to be represented as a relation.
            arity (int = 2): the number of elements within one tuple of the relation. This should be at
                least 0, which would be the empty relation.

        Attributes: 
            rList (list of (tuple of(int))): the relation represented as a list of tuples of integers.
            n (int): the length of the n-ary image to be represented as a relation.
        """

        if isGraph:
            self.rList = [(x, y) for x in range(len(graphOrList[0])) for y in range(len(graphOrList[0])) if graphOrList[x][y] == 1]
        else:
            self.rList = graphOrList
        
        self.n = n
        self.arity = arity



def randomPermutationMatrix(n):
    #n: integer
    columnSelection = [i for i in range(n)]
    permutationMatrix = np.zeros((n,n))
    trueOrFalse = [True, False]
    for i in range(n):
        IndexSelection = random.choice(columnSelection)
        columnSelection.remove(IndexSelection)
        ParitySelection = random.choice(trueOrFalse)
        permutationMatrix[i, IndexSelection] = 1 if ParitySelection else -1
    return permutationMatrix

def applyRandomPermutation(n, coords):
    
    # n: length of side
    # coords: array of tuples
    dim = len(coords[0])
    randomPermutation = randomPermutationMatrix(dim)
    print("ur mom", randomPermutation)
    newRelationRList = []
    for coordinate in coords:
        column_vector = np.array([[l] for l in coordinate])
        translationVector = np.array([[(n-1)/2] for l in range(dim)])
        result = np.add(np.matmul(randomPermutation, np.subtract(column_vector, translationVector)), translationVector)
        result_tuple = tuple(result.flatten())  # Convert the NumPy array to a tuple
        newRelationRList.append(result_tuple)
    return Relation(newRelationRList, False, n, dim)
    """
    return({
        "result": np.add(np.matmul(randomPermutation, np.subtract(column_vector, translationVector)), translationVector),
        "permutationMatrix": randomPermutation
    })
    """


class HyperoctohedralAutomorphism(Operation):
    """
    An automorphism of the Hamming graph obtained by applying a random permutation of the hyperoctohedral group.
    """

    def __init__(self):
        """
        Create a hyperoctohedral automorphism.

        """
        
        Operation.__init__(self, 1, lambda x: applyRandomPermutation(x.n, x.rList), cache_values=False)




def dot_product(x, y):
    """
    Take the dot product of two n-nary images over the finite field of order 2.

    Arguments:
        x (Relation): The first n-nary image.
        y (Relation): The second n-nary image.
    Returns:
        int: The dot product of the given images.
    """

    return len([tup for tup in x.rList if tup in y.rList]) % 2




def indicator_polymorphism(tup, a, c):
    """
    Perform an indicator polymorphism where the either blank (all zeroes) or the n-ary image with a
    single black pixel at position of the given tuple.

    Args:
        tup (tuple): the location of the single black pixel
        a (iterable of Relation): A sequence of n-nary images with which dot products are to be taken.
        c (iterable of Relation): A sequence of n-nary images with which dot products are to be taken.

    Returns:
        Relation: The n-ary image obtained by applying the indicator polymorphism.
    """

    if all(dot_product(a[k], c[k]) for k in range(len(a))):
        return Relation([tup], False, a[0].n, a[0].arity)
    return Relation([], False, a[0].n, a[0].arity)



class IndicatorPolymorphism(Operation):
    """
    Create a polymorphism of the Hamming graph by taking dot products with fixed n-ary images.
    """

    def __init__(self, tup, c):
        """
        Create an indicator polymorphism where the output image is either blank (all zeroes) or the n-ary image with a
        single black pixel at position given by the tuple.

        Args:
            tup (tuple): the location of the single black pixel
            c (iterable of Relation): A sequence of n-ary images with which dot products are to be
                taken.
        """

        Operation.__init__(self, len(c), lambda a: indicator_polymorphism(tup, a, c), cache_values=False)



def hamming_distance(x, y):
    """
    Compute the Hamming distance between a pair of n-ary images.

    Args:
        x (Relation): the first n-ary image
        y (Relation): the second n-ary image

    Returns:
        int: the Hamming distance between x and y, ie the number of pixels at which they differ.

    """
    return len([tup for tup in x.rList if tup not in y.rList]) + len([tup for tup in y.rList if tup not in x.rList])



def swapping(x, y):
    """
    Taking the component-wise sum of two n-ary images.

    Args:
        x (Relation): The first n-ary image to be added
        y (Relation): The second n-ary image to be added

    Returns:
        Relation: The image with entries of componentwise sums of x and y
    
    """

    return Relation([tup for tup in x.rList if tup not in y.rList] + [tup for tup in y.rList if tup not in x.rList], 
                     False, x.n, x.arity)




class SwappingAutomorphism(Operation): 
    """
    An automorphism of the Hamming graph obtained by taking the componentwise sum with a fixed n-ary image.
    """

    def __init__(self, b):
        """
        Create a swapping automorphism for a given image.

        Argument:
            b (Relation): The fixed n-ary image used for swapping pixels, represented as a relation
        """

        Operation.__init__(self, 1, lambda a: swapping(a, b),
                           cache_values=False)



def blanking(x, y):
    """
    Taking the Hadamard product (component-wise multiplication) of two n-ary images.

    Args:
        x (Relation): The first n-ary image in the Hadamard product
        y (Relation): The second n-ary image in the Hadamard product

    Returns:
        Relation: The image of Hadamard product of x and y
    
    """

    return Relation([tup for tup in x.rList if tup in y.rList], False, x.n, x.arity)




class BlankingEndomorphism(Operation):
    """
    An endomorphism of the Hamming graph obtained by taking the Hadamard product with a fixed n-ary image.
    """

    def __init__(self, b):
        """
        Create a blanking endomorphism for a given image.

        Argument:
            b (Relation): The fixed n-ary image used for swapping pixels, represented by relation.
        """


        Operation.__init__(self, 1, lambda a: swapping(a, b), cache_values=False)




def polymorphism_neighbor_func(op, num_of_neighbors, constant_images):
    """

    Arguments:
        op (Operation): A n-ary image polymorphism operation.
        num_of_neighbors (int): The amount of possible neighbors to generate.
        constant_images (list of Relation): A list of n-ary images (as relations) to use as constants.

    Returns:

    """
    

    endomorphisms = []
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


#print(dot_product(a, a2))
#print(indicator_polymorphism((1, 1), [a, a2], [a, a2]))
#print(hamming_distance(a, a2))
#print(swapping(a, a2))
#print(blanking(a, a2))

#newRelation = applyRandomPermutation(3, (2, 1, 0))
#print(newRelation.rList, newRelation.n)
