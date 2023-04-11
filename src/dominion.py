"""
Dominion

Tools for creating 2-dimensional dominions
"""

import random
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import numpy as np
import math
import copy


class Graph:
    def _init_(self):
        self.vertices = []
        self.edges = {}

    def addVertex(self, vertexName):
        if hasattr(self, "vertices"):
            self.vertices.append(vertexName)
        else:
            self.vertices = [vertexName]

    # making this an unordered graph so that from any label we can look up the adjacent labels
    def addEdge(self, vertex1, vertex2):
        if hasattr(self, "edges"):
            if vertex1 in self.edges:
                self.edges[vertex1].add(vertex2)
            else:
                self.edges.update({vertex1: {vertex2}})
            if vertex2 in self.edges:
                self.edges[vertex2].add(vertex1)
            else:
                self.edges.update({vertex2: {vertex1}})
        else:
            self.edges = {vertex1: {vertex2}, vertex2: {vertex1}}

    def getNeighbors(self, vertex1):
        return list(self.edges[vertex1])

    def __str__(self):
        vertices = ""
        edges = ""
        if hasattr(self, "vertices"):
            vertices = self.vertices
        if hasattr(self, "edges"):
            edges = self.edges
        return f'Vertices: {vertices}\nEdges: {edges}'

    def setRoot(self, vertex):
        self.root = vertex


def new_row(row, set_of_labels, constraint_graph=None):
    """
    Construct a new row for a dominion with a given set of labels and a graph constraining
    which labels can appear together.
    Args:
        row (list): A list of labels representing a row of a dominion.
        set_of_labels (set): The pixel labels used in the dominion. The entries of `row` should come from this set.
        constraint_graph (Graph): The graph determining which labels can appear next to each other.
            The vertices of `constraint_graph` should be the members of `set_of_labels`. The default value None
            behaves as though the graph is the complete graph on the vertex set `set_of_labels'.
    Returns:
        list: A new row which is permitted to follow `row` in a dominion with the given labels and constraints.
    """

    partial_row = []
    n = len(row)
    for i in range(n):
        if i == 0:
            candidates = {row[0], row[1]}
            rejects = set([])
        elif i == n - 1:
            candidates = {row[n - 2], row[n - 1], partial_row[n - 2]}
            rejects = set([])
        else:
            candidates = {row[i - 1], row[i], partial_row[i - 1]}.intersection({row[i], row[i + 1]})
            rejects = {row[i - 1], row[i], row[i + 1], partial_row[i - 1]}.difference(candidates)
        # If there are 2 candidate labels then we are done, since we know the only two possible choices for
        # the new label. If there is only 1 candidate then either there are some rejects, in which case it must be
        # chosen, or there are no rejects, in which case a new label may appear. This new label is subject to the
        # adjacency relation for `constraint_graph`.
        if len(candidates) == 1 and len(rejects) == 0:
            if constraint_graph is None:
                candidates = set_of_labels
            else:
                candidates = candidates.union(constraint_graph.getNeighbors(list(candidates)[0]))
                # may need to test if getNeighbors works for this case, the code wasn't reaching it when I tested it

        partial_row += (random.sample(list(candidates), 1))
    return partial_row


def random_tree(set_of_labels):
    """
    Construct a random tree on a given set of labels.

    Args:
        set_of_labels (iterable): The vertex labels for the tree.
    Returns:
        Graph: A randomly-chosen tree on the given vertices.
    """

    labels = list(set_of_labels)

    G = Graph()

    while len(labels) > 0:
        if not hasattr(G, "vertices"):
            node = labels.pop()
            G.addVertex(node)
        else:
            node = random.choice(G.vertices)
            for _ in range(random.randint(1, len(labels))):
                neighbor = labels.pop()
                G.addVertex(neighbor)
                G.addEdge(node, neighbor)
    return G


def random_dominion(size, set_of_labels, constraint_graph=None):
    """

    """

    if constraint_graph is None:
        partial_dominion = [random.choices(tuple(set_of_labels), k=size)]
    else:
        partial_dominion = [[random.choice(tuple(set_of_labels))]]
        for _ in range(1, size):
            # print("reached getNeighbor case")#delete
            partial_dominion[0].append(
                random.choice(constraint_graph.getNeighbors(partial_dominion[0][-1])))  # test getNeighbors
    for _ in range(size - 1):
        partial_dominion.append(new_row(partial_dominion[-1], set_of_labels, constraint_graph))
    return partial_dominion


def draw_dominion(dominion, colmap, name):
    """
    Render an image from a given dominion and color map.
    Args:
        dominion (list): A list encoding a dominion.
        colmap (string): The name of a color map.
        name (string): The name of the resulting file.
    """

    plt.imsave('{}.png'.format(name), np.array(dominion), cmap=eval('cm.{}'.format(colmap)))


# reads a dominion that's been stored as a file and returns a dominon as an np array
def readDominion(file, n):
    fileText = file.read()
    dominion = np.zeros((3, 3))
    n = 0
    for i in range(0, 3):
        for j in range(0, 3):
            while not list(fileText)[n].isdigit():
                n += 1
            dominion[i][j] = list(fileText)[n]
            n += 1

    return dominion


# --------------------functions for creating activation functions------------------------------


# runs psi_k by counting the number of black squares (squares with value 1)
def psiK(image1, image2, n):
    images = [image1, image2]
    counter = [0, 0]
    for i in range(0, 2):
        for j in range(0, n):
            for k in range(0, n):
                if images[i][j][k] == 1:
                    counter[i] += 1
    return counter


# applies a dominion to given Hamming weights
def applyDominion(D, weights):
    return D[weights[0]][weights[1]]


# returns a random nxn binary image
def drawImage(n):
    image = np.empty((n, n))
    for i in range(0, n):
        for j in range(0, n):
            image[i][j] = random.randint(0, 1)
    return image


# returns a random image that's adjacent to the input image in the Hamming graph (could return original image) by switching the value of at most 1 pixel
def randomAdjacent(image):
    # print(image)    #delete
    n = image.shape[1]
    digit = random.randint(0, n * n)
    print(digit)  # delete
    if digit == n * n:
        return image
    else:
        # convert digit to pixel position:
        x1 = math.floor(digit / n)
        x2 = digit % n
        # print(str(x1)+", "+str(x2))    #delete
        newImage = copy.deepcopy(image)
        newImage[x1][x2] = abs(image[x1][x2] - 1)
        print(newImage)  # delete
        return newImage


# runs recursion step for getHomomorphism
# *****in progress*****
# have each step modify then return alpha
def getHelper(L, root, neighbors, alpha):
    print("root: " + str(root))  # delete
    # print(neighbors)    #delete
    if neighbors == None:
        return alpha
    else:
        for node in neighbors:
            print(node)
            alpha.update({int(node): randomAdjacent(alpha.get(int(root)))})
            newNeighbors = L.getNeighbors(node)
            newNeighbors.remove(root)
            alpha = getHelper(L, node, newNeighbors, alpha)
            # print(alpha)    #delete
    return alpha


# This should give the homomorphism from tree L to Ham_n
# currently explores nodes in the correct order but doesn't associate them with anything in Hamming graph yet
# *****in progress*****
def getHomomorphism(L, n):
    r = L.root
    alpha = {int(r): drawImage(n)}
    print(alpha.get(int(r)))  # delete
    print(alpha)  # delete
    return getHelper(L, r, L.getNeighbors(r), alpha)


# creates g_alpha by composing psi, D, and alpha
# Does D function on the output of psiK or on the Hamming weight graph?
# *****in progress*****
n = 3


def getGAlpha(image1, image2):
    def _new_fnc():
        weights = psiK(image1, image2, n)
        return

    return _new_fnc()


# --------------------test and run functions------------------------------


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

alpha = getHomomorphism(L, 2)
print(alpha)

"""
#CODE THAT RUNS OR TESTS EXISTING FUNCTIONS:
#Creates a random tree and generates 20 dominions. Currently this is set so you manually update treeNumber each time
# you make a new tree
labels={1, 2, 3, 4, 5}
L=random_tree(labels)
print(L)
treeNumber=0
for i in range (0, 20):
    fileName="Dominions\Tree"+str(treeNumber)+"-Dominion"+str(i)+".txt"
    dominionFile=open(fileName, "w")
    dominionFile.write(str(random_dominion(3,labels, L)))
    dominionFile.close()

#Creates a random tree given a list of labels and tests the getNeighbors function
testList={"test1", "test2", "test3", "test4", "test5","test6"}
tree=random_tree(testList)
print(tree)
print(tree.getNeighbors('test1'))
#This chunk of code is to test applyDominion
#runs dominion reader
file=open(r"Dominions\Tree0-Dominion4.txt", "r")
dominion=readDominion(file, 3)
#constructs two 3x3 images and runs psi_k
image1=[[0, 0, 1],[0, 0, 0],[1, 0, 0]]
image2=[[0, 0, 0],[0, 0, 0],[0, 1, 0]]
weights=psiK(image1, image2, 3)
print(applyDominion(dominion, weights))         #row 2, column 1 (indexed from 0)
file.close()
"""

# Note: every 3x3 dominion with label set size 3 had at least one 3 in - is this a feature of the program or just
# coincidence? - also random trees with three labels kept generating same tree

# still to code:
# reconstruct tree from file
# construct alpha from file
# alpha:
# store alpha as a file
# s_alpha
# store g_alpha

# adjustments to code:
# dominions need to be large enough to work with any nxn image (ie for 3x3 images dominion should be 10x10)
# store a folder for each tree with tree file and folders for dominions and homomorphisms
