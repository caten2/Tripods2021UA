import numpy as np
import random
# Key:
"""
T = Translation
O = Origin
R = Rotation
Int = Angle
"""
"""
class Relation:

    def cleanData(self, graphOrList):

        return []

    def __init__(self, graphOrList, isGraph, n, arity=2):
        
        #Parameters: graphOrList, List<(int, int)> or List<List><int> & isGraph, boolean,  arity, int default 2
        #Attributes: rList, List<(int, int)>
        

        if isGraph:
            self.rList = [(x, y) for x in range(len(graphOrList[0])) for y in range(len(graphOrList[0])) if graphOrList[x][y] == 1]
        else:
            self.rList = graphOrList
        self.n = n
"""


def TOR90TOPermutation(n, coords):
    # Input type: n: integer
    # Coords, Tuple of 3 coordinates x, y, z
    array = np.array([x for x in coords])

    return(np.add(np.matmul(np.array([
                        [0, -1, 0],
                        [1, 0, 0],
                        [0, 0, 1]]), np.subtract(array, np.array([(n-1)/2, (n-1)/2, (n-1)/2]))), np.array([(n-1)/2, (n-1)/2, (n-1)/2])))
    

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
    # coords: tuple of relation
    dim = len(coords)
    randomPermutation = randomPermutationMatrix(dim)
    column_vector = np.array([l for l in coords])
    translationVector = np.array([(n-1)/2 for l in range(dim)])
    result = [tuple(np.add(np.matmul(randomPermutation, np.subtract(column_vector, translationVector)), translationVector))]
    return Relation(result, False, n, dim)
    """
    return({
        "result": np.add(np.matmul(randomPermutation, np.subtract(column_vector, translationVector)), translationVector),
        "permutationMatrix": randomPermutation
    })
    """


    


newRelation = applyRandomPermutation(3, (2, 1, 0))
print(newRelation.rList, newRelation.n)

#TOR90TOPermutation(4, (2, 1, 0))

#randomPermutationMatrix(4)


