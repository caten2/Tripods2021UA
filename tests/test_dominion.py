"""
Tests for dominion polymorphisms
"""

from dominion import *
from neural_net import Layer, Neuron
from polymorphisms import RotationAutomorphism, ReflectionAutomorphism, SwappingAutomorphism, BlankingEndomorphism, \
    IndicatorPolymorphism
from mnist_training_binary import binary_mnist_zero_one

# --------------- Testing Basic Functionalities ---------------


# Thanks to the implementation of relations as sets of tuples, the functions should work the same regardless of arity.

# The sizes of the universe of the relations and the appropriate dominions.
size1 = 2
dominion_size1 = size1 ** 2 + 1
print('The size of the universe:', size1)
print('The size of the appropriate dominions:', dominion_size1)
print('We are using the labels 0, 1, and 2 for dominions. Label i corresponds to a relation containing i tuples.')


# Assigning a relation to each label. In this test, the set of labels is assumed to be {0, 1, 2}.
def relabeling1(i):
    if i == 0:
        return Relation([], 2, 2)
    elif i == 1:
        return Relation([(0, 0)], 2, 2)
    else:
        return Relation([(0, 0), (1, 1)], 2, 2)


# A pair of relations to use as test inputs.
relations1 = (Relation([], 2, 2),
              Relation([], 2, 2))

# Generating a random dominion and printing it in matrix form.
dominion1 = random_dominion(dominion_size1, range(3), None)
print('Random dominion:')
print('\n'.join(map(str, dominion1)))

# The polymorphism associated with dominion1, with respect to the relabeling1.
poly1 = DominionPolymorphism(dominion1, relabeling1)
print('The result of applying the corresponding polymorphism to a pair of empty relations:')
print('-', poly1(*relations1))
print('This relation corresponds to the label at (0, 0), namely {0}.'.format(dominion1[0][0]))

print('\n{0}\n'.format('-' * 60))

# TODO --------------- Testing on Neural Nets: MULTIPLICATION ---------------


# TODO --------------- Testing on Neural Nets: MNIST ---------------


# The dimension of an MNIST image.
mnist_image_size = 28
# The size of dominions used for MNIST images.
mnist_dominion_size = mnist_image_size ** 2 + 1
# The labels used for dominions.
# The label 0 means an empty relation. A label with pair x, y (possibly x == y) of pairs is the relation that
# contains x and y.
mnist_labels = [0] + [((x1, y1), (x2, y2))
                      for x1 in range(mnist_image_size)
                      for y1 in range(mnist_image_size)
                      for x2 in range(mnist_image_size)
                      for y2 in range(mnist_image_size)]


# The relabeling function.
def mnist_relabeling(i):
    if i == 0:
        return Relation([], 28, 2)
    x, y = i
    return Relation([x, y], 28, 2)


# Load some binary images from the modified MNIST training set.
mnist_training_pairs = tuple(binary_mnist_zero_one(100, 'train'))

# Constructing a 3-layer neural network that takes 2 MNIST images of 0s or 1s, trained as follows:
# - If both relations represent the same digit the output is the full relation.
# - Otherwise, the output is the empty relation.

# First layer has 2 inputs.
mnist_layer0 = Layer(('x0', 'x1'))

# Second layer has 2 neurons.
mnist_op0 = DominionPolymorphism(random_dominion(mnist_dominion_size, mnist_labels, None), mnist_relabeling)
mnist_op1 = DominionPolymorphism(random_dominion(mnist_dominion_size, mnist_labels, None), mnist_relabeling)
mnist_node0 = Neuron(mnist_op0, ('x0', 'x1'))
mnist_node1 = Neuron(mnist_op1, ('x0', 'x1'))
mnist_layer1 = [mnist_node0, mnist_node1]

# TODO --------------- Leftover Testing Code ---------------


"""
#CODE THAT RUNS OR TESTS EXISTING FUNCTIONS:


#Creates a random tree and generates 20 dominions and 20 homomorphisms. Currently this 
is set so you manually update treeNumber each time you make a new tree
labels={1, 2, 3, 4, 5}
L=random_tree(labels)
print(L)
treeNumber=3
L.makeTreeFile(treeNumber)
#print(readTree("Tree"+str(treeNumber)+"\\Tree"+str(treeNumber)+".txt"))
generateDominions(L, labels, treeNumber, 20)
L=readTree("Tree"+str(treeNumber)+"\\Tree"+str(treeNumber)+".txt")
L.setRoot(1)
generateHomomorphisms(L, treeNumber, 20, 28)



# Creates a tree L for testing getHomomorphism
L = Graph()
for i in range(1, 9):
    L.addVertex(i)
L.addEdge(1, 2)
L.addEdge(1, 3)
L.addEdge(2, 4)
L.addEdge(2, 5)
L.addEdge(2, 6)
L.addEdge(5, 7)
L.addEdge(5, 8)
L.setRoot(1)

print(L)




#Creates a random tree given a list of labels and tests the getNeighbors function
testList={"test1", "test2", "test3", "test4", "test5","test6"}
tree=random_tree(testList)
print(tree)
print(tree.getNeighbors('test1'))



#This chunk of code is to test applyDominion
#runs dominion reader
file=open(r"Dominions\\Tree0-Dominion4.txt", "r")
dominion=readDominion(file)

#constructs two 3x3 images and runs psi_k
image1=[[0, 0, 1],[0, 0, 0],[1, 0, 0]]
image2=[[0, 0, 0],[0, 0, 0],[0, 1, 0]]
weights=psiK(image1, image2)

print(applyDominion(dominion, weights))         #row 2, column 1 (indexed from 0)


file.close()
"""

# Note: every 3x3 dominion with label set size 3 had at least one 3 in in - is this a
# feature of the program or just coincidence? - also random trees with three labels
# kept generating same tree

# check out labels var in generateDominions - substituting tree.vertices would
# probably work
# figure out a better way to get n as an input in _func

# Remember that dominions need to be large enough to work with any nxn image (eg for
# 3x3 images dominion should be 1
