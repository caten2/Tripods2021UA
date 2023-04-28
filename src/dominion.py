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
from discrete_neural_net import Operation
from numpy import array


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

    def makeTreeFile(self, treeNumber):
            fileName="Tree"+str(treeNumber)+"\Tree"+str(treeNumber)+".txt"
            treeFile=open(fileName, "w")


            vertices=""
            edges=""
            if hasattr(self,"vertices"):
                vertices=self.vertices
            if hasattr(self, "edges"):
                edges=self.edges

            treeFile.write(f'{vertices}\n{edges}')

            treeFile.close()


class GAlpha(Operation):

    def __init__(self, dominion, alpha):
        def _func(images):
            #images should be a tuple of images
            weights=psiK(images[0], images[1])
            temp=dominion(weights)
            return alpha(temp)

        Operation.__init__(self, 2, _func, False)

    #making this more specific (the generic parent version is probably better but it was giving me errors so I'm writing this for now)
    #errors are because of index type so I changed __init__ to not store values, probably want to change back
    def __getitem__(self, index):
        """
        Compute the value of the Operation on given inputs.

        Argument:
            index (tuple): The tuple of inputs to plug in to the Operation.
        """
        #print(index)    #delete

        if self.cache_values:
            if self.arity == 0:
                if not self.values:
                    self.values = self.func(index[0],index[1])
            if self.arity > 0:
                if tuple(index) not in tuple(self.values.keys()):
                    #self.values is a dictionary where the key is the input to the function and the value is the function output
                    #error is definitely with index as input to self.values - currently index is a tuple but the elements of the tuple are images, which are lists of lists, which are unhashable and thus can't be keys in dictionaries
                    self.values[tuple(index)] = self.func(index[0],index[1])
            return self.values[tuple(index)]
        return self.func([index[0],index[1]])




    


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
            partial_dominion[0].append(
                random.choice(constraint_graph.getNeighbors(partial_dominion[0][-1])))
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

"""
# reads a dominion that's been stored as a file and returns a dominon as an np array
#I think this might hard code a 3x3 dominion
def readDominion(fileName):
    file=open(fileName, "r")
    fileText = file.read()
    dominion = np.zeros((3, 3))
    n = 0
    for i in range(0, 3):
        for j in range(0, 3):
            while not list(fileText)[n].isdigit():
                n += 1
            dominion[i][j] = list(fileText)[n]
            n += 1
    file.close()

    return dominion
    """


def readTree(fileName):
    file=open(fileName, "r")
    tree=Graph()
    line1=file.readline()
    for i in range(0, len(line1)):
        if line1[i].isdigit():
            tree.addVertex(int(line1[i]))

    line2=file.readline()
    line2=line2.split("}")

    list=[]
    for item in line2:
        item=item.replace(",","")
        item=item.replace(" ","")
        list.append(item.replace("{",""))

    for item in list:
        if not item=="":
            index1=item[0]
            for j in range(2, len(item)):
                tree.addEdge(int(index1),int(item[j]))

    file.close()

    return tree


# --------------------functions for creating activation functions------------------------------


# runs psi_k by counting the number of black squares (squares with value 1)
#assumes that we're using a 28x28 image
def psiK(image1, image2):
    images = [image1, image2]
    counter = [0, 0]
    for i in range(0, 2):
        for j in range(0, 28):
            for k in range(0, 28):
                if images[i][j][k] == 1:
                    counter[i] += 1
    return counter


# applies a dominion to given Hamming weights
def applyDominion(D, weights):
    return D[weights[0]][weights[1]]


# returns a random dim*dim binary image (dimension dim)
def drawImage(dim):
    image = np.empty((dim, dim))
    for i in range(0, dim):
        for j in range(0, dim):
            image[i][j] = random.randint(0, 1)
    return image


# returns a random image that's adjacent to the input image in the Hamming graph (could return original image) by switching the value of at most 1 pixel
def randomAdjacent(image):
    n = image.shape[1]
    digit = random.randint(0, n * n)
    #print(digit)  # delete
    if digit == n * n:
        return image
    else:
        # convert digit to pixel position:
        x1 = math.floor(digit / n)
        x2 = digit % n
        newImage = copy.deepcopy(image)
        newImage[x1][x2] = abs(image[x1][x2] - 1)
        #print(newImage)  # delete
        return newImage


# runs recursion step for getHomomorphism
# have each step modify then return alpha
def getHelper(L, root, neighbors, alpha):
    #print("root: " + str(root))  # delete
    if neighbors == None:
        return alpha
    else:
        for node in neighbors:
            #print(node)
            alpha.update({int(node): randomAdjacent(alpha.get(int(root)))})
            newNeighbors = L.getNeighbors(node)
            newNeighbors.remove(root)
            alpha = getHelper(L, node, newNeighbors, alpha)
            # print(alpha)    #delete
    return alpha


# This creates a homomorphism from tree L to Ham_n
def getHomomorphism(L, n):
    r = L.root
    alpha = {int(r): drawImage(n)}
    #print(alpha.get(int(r)))  # delete
    #print(alpha)  # delete
    return getHelper(L, r, L.getNeighbors(r), alpha)





def dominionToFnc(dominion):
    return lambda weights: dominion[weights[0]][weights[1]]
        



def alphaTextToFnc(alphaText):
    alphaDict=eval(alphaText)
    return lambda node: alphaDict[int(node)]




def getGAlpha(treeNum):
    #Pick random dominion and homomorphism
    fileNameD="Tree"+str(treeNum)+"\Dominions\Tree"+str(treeNum)+"-Dominion"+str(random.randint(0, 19))+".txt"
    fileNameH="Tree"+str(treeNum)+"\Homomorphisms\Tree"+str(treeNum)+"-Homomorphism"+str(random.randint(0, 19))+".txt"
    dominionFile=open(fileNameD, "r")
    homomorphismFile=open(fileNameH, "r")
    dominion=eval(dominionFile.read())

    
    dominionFnc=dominionToFnc(dominion)
    alphaFnc=alphaTextToFnc(homomorphismFile.read())
    

    gAlpha=GAlpha(dominionFnc, alphaFnc)


    homomorphismFile.close()

    return gAlpha



# --------------------test and run functions------------------------------

#Given a tree, generates a set number of dominions and stores them as text files. Note that a folder with the corresponding treeNum must already be set up
def generateDominions(tree, labels, treeNum, dominionNum):

    for i in range (0, dominionNum):
        fileName="Tree"+str(treeNum)+"\Dominions\Tree"+str(treeNum)+"-Dominion"+str(i)+".txt"
        dominionFile=open(fileName, "w")

        dominionFile.write(str(random_dominion(28*28+1,labels, tree)))

        dominionFile.close()


#Given a tree, generates a set number of homomorphsisms and stores them as text files. Note that a folder with the corresponding treeNum must already be set up. n is the dimension of the images
def generateHomomorphisms(tree, treeNum, homomNum, n):

    for i in range (0, homomNum):
        fileName="Tree"+str(treeNum)+"\Homomorphisms\Tree"+str(treeNum)+"-Homomorphism"+str(i)+".txt"
        homomFile=open(fileName, "w")

        homomFile.write(str(getHomomorphism(tree, n)))

        homomFile.close()








"""
#CODE THAT RUNS OR TESTS EXISTING FUNCTIONS:


#Creates a random tree and generates 20 dominions and 20 homomorphisms. Currently this is set so you manually update treeNumber each time you make a new tree
labels={1, 2, 3, 4, 5}
L=random_tree(labels)
print(L)
treeNumber=3
L.makeTreeFile(treeNumber)
#print(readTree("Tree"+str(treeNumber)+"\Tree"+str(treeNumber)+".txt"))
generateDominions(L, labels, treeNumber, 20)
L=readTree("Tree"+str(treeNumber)+"\Tree"+str(treeNumber)+".txt")
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
file=open(r"Dominions\Tree0-Dominion4.txt", "r")
dominion=readDominion(file)

#constructs two 3x3 images and runs psi_k
image1=[[0, 0, 1],[0, 0, 0],[1, 0, 0]]
image2=[[0, 0, 0],[0, 0, 0],[0, 1, 0]]
weights=psiK(image1, image2)

print(applyDominion(dominion, weights))         #row 2, column 1 (indexed from 0)


file.close()
"""

#Note: every 3x3 dominion with label set size 3 had at least one 3 in in - is this a feature of the program or just coincidence? - also random trees with three labels kept generating same tree

    #check out labels var in generateDominions - susbstituting tree.vertices would probably work
    #figure out a better way to get n as an input in _func

#Remember that dominions need to be large enough to work with any nxn image (eg for 3x3 images dominion should be 10x10)
