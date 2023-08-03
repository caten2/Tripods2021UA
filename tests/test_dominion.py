# TODO: Make this into a test script for dominion.py.
# --------------------test and run functions------------------------------

# The dimension of an MNIST image.
mnist_image_size = 28
# The size of dominions used for MNIST images.
dominion_size_mnist = mnist_image_size ** 2 + 1


# TODO: The `28` is hardcoded appropriately for MNIST data. Introduce a constant that
#  holds this value and explain its purpose.
# Given a tree, generates a set number of dominions and stores them as text files. Note
# that a folder with the corresponding tree_num must already be set up
def generate_dominions(tree, labels, tree_num, dominion_num):
    for i in range(dominion_num):
        file_name = "Tree" + str(tree_num) + "\\Dominions\\Tree" + str(
            tree_num) + "-Dominion" + str(i) + ".txt"
        dominion_file = open(file_name, "w")

        dominion_file.write(str(random_dominion(dominion_size_mnist, labels, tree)))

        dominion_file.close()


# TODO: Better documentation.
# Given a tree, generates a set number of homomorphisms and stores them as text files.
# Note that a folder with the corresponding tree_num must already be set up. n is the
# dimension of the images.
def generate_homomorphisms(tree, tree_num, hom_num, n):
    for i in range(0, hom_num):
        file_name = "Tree" + str(tree_num) + "\\Homomorphisms\\Tree" + str(
            tree_num) + "-Homomorphism" + str(i) + ".txt"
        hom_file = open(file_name, "w")

        hom_file.write(str(get_homomorphism(tree, n)))

        hom_file.close()


# TODO: Put these test methods in a separate test_dominion.py script.
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
file=open(r"Dominions\Tree0-Dominion4.txt", "r")
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