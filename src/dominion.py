"""
Dominion

Tools for creating 2-dimensional dominions
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from discrete_neural_net import Operation


# --------------- Classes -----------------


class Graph:
    def __init__(self):
        self.root = None
        self.vertices = []
        self.edges = {}

    def add_vertex(self, vertex_name):
        if hasattr(self, "vertices"):
            self.vertices.append(vertex_name)
        else:
            self.vertices = [vertex_name]

    def add_edge(self, vertex1, vertex2):
        """
        Adds edges (vertex1, vertex2) and (vertex2, vertex1) to the graph.
        """
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

    def get_neighbors(self, vertex1):
        return list(self.edges[vertex1])

    def __str__(self):
        vertices = ""
        edges = ""
        if hasattr(self, "vertices"):
            vertices = self.vertices
        if hasattr(self, "edges"):
            edges = self.edges
        return f'Vertices: {vertices}\nEdges: {edges}'

    def set_root(self, vertex):
        self.root = vertex

    def make_tree_file(self, tree_num):
        file_name = file_name_tree(tree_num)
        tree_file = open(file_name, "w")

        vertices = ""
        edges = ""
        if hasattr(self, "vertices"):
            vertices = self.vertices
        if hasattr(self, "edges"):
            edges = self.edges

        tree_file.write(f'{vertices}\n{edges}')

        tree_file.close()


class DominionPolymorphism(Operation):

    def __init__(self, dominion, alpha):
        def _func(*images):
            # images should be a tuple of images
            weights = (hamming_weight(images[0]), hamming_weight(images[1]))
            temp = dominion(weights)
            return alpha(temp)

        Operation.__init__(self, 2, _func, False)


# --------------- Generating Dominions --------------


def new_row(row, set_of_labels, constraint_graph=None):
    """
    Construct a new row for a dominion with a given set of labels and a graph constraining
    which labels can appear together.
    Args:
        row (list): A list of labels representing a row of a dominion.
        set_of_labels (set): The pixel labels used in the dominion. The entries of
        `row` should come from this set.
        constraint_graph (Graph): The graph determining which labels can appear next to
        each other.
            The vertices of `constraint_graph` should be the members of
            `set_of_labels`. The default value `None`
            behaves as though the graph is the complete graph on the vertex set
            `set_of_labels'.
    Returns:
        list: A new row which is permitted to follow `row` in a dominion with the given
        labels and constraints.
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
            candidates = {row[i - 1], row[i], partial_row[i - 1]}.intersection(
                {row[i], row[i + 1]})

            rejects = {row[i - 1], row[i], row[i + 1], partial_row[i - 1]}.difference(
                candidates)

        # If there is a single candidate and no rejects, we can choose any neighboring label of the candidate.
        if len(candidates) == 1 and len(rejects) == 0:
            if constraint_graph is None:
                candidates = set_of_labels
            else:
                candidates = candidates.union(
                    constraint_graph.get_neighbors(list(candidates)[0]))

        # Add a random candidate.
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

    tree = Graph()
    node = labels.pop()
    tree.add_vertex(node)

    while len(labels) > 0:
        node = random.choice(tree.vertices)
        for _ in range(random.randint(1, len(labels))):
            neighbor = labels.pop()
            tree.add_vertex(neighbor)
            tree.add_edge(node, neighbor)
    return tree


def random_dominion(size, set_of_labels, constraint_graph=None):
    """
    Generates a random dominion given its size, set of labels, and constraint graph.
    """

    if constraint_graph is None:
        partial_dominion = [random.choices(tuple(set_of_labels), k=size)]
    else:
        partial_dominion = [[random.choice(tuple(set_of_labels))]]
        for _ in range(1, size):
            partial_dominion[0].append(
                random.choice(constraint_graph.get_neighbors(partial_dominion[0][-1])))
    for _ in range(size - 1):
        partial_dominion.append(
            new_row(partial_dominion[-1], set_of_labels, constraint_graph))
    return partial_dominion


# --------------- Creating Polymorphisms ---------------


def hamming_weight(image):
    """
    Returns the Hamming weight of a 2D binary image.
    """
    weight = sum([sum(row) for row in image])
    return weight


def random_image(dim):
    """
    Returns a random binary image of size d * d.
    """
    image = np.empty((dim, dim))
    for i in range(0, dim):
        for j in range(0, dim):
            image[i][j] = random.randint(0, 1)
    return image


def random_adjacent_image(image):
    """
    Returns a random neighbor of a given binary image in the Hamming graph by switching the value of at most 1 pixel.
    """
    n = image.shape[1]
    pixel_i = random.randint(0, n - 1)
    pixel_j = random.randint(0, n - 1)
    if pixel_i == n and pixel_j == n:
        return image
    else:
        new_image = copy.deepcopy(image)
        new_image[pixel_i][pixel_j] = 1 - image[pixel_i][pixel_j]
        return new_image


def get_homomorphism(tree, n):
    """
    Returns a homomorphism  from a tree to the Hamming graph H_n.
    """
    vertex = tree.root
    neighbors = tree.get_neighbors(vertex)
    hom = {int(vertex): random_image(n)}

    while neighbors:
        new_vertex = neighbors[0]
        neighbors = tree.get_neighbors(new_vertex)
        neighbors.remove(vertex)

        hom.update({int(new_vertex): random_adjacent_image(hom.get(int(vertex)))})
        vertex = new_vertex

    return hom


# --------------- IO -----------------


def file_name_tree(tree_num):
    file_name = "Tree" + str(tree_num) + "\\Tree" + str(tree_num) + ".txt"
    return file_name


def file_name_dominion(tree_num, dom_num):
    file_name = "Tree" + str(tree_num) + "\\Dominions\\Tree" + str(tree_num) + "-Dominion" + str(dom_num) + ".txt"
    return file_name


def file_name_homomorphism(tree_num, hom_num):
    file_name = "Tree" + str(tree_num) + "\\Dominions\\Tree" + str(tree_num) + "-Dominion" + str(hom_num) + ".txt"
    return file_name


def read_dominion(file_name):
    """
    Reads a dominion from a file and returns the dominion as an `np` array.

    Args:
        file_name: The path of the file to be read from. It has to contain the string representation of a dominion.
    """
    file = open(file_name, "r")
    file_text = file.read()
    dominion = np.asarray(eval(file_text))
    file.close()
    return dominion


# TODO: See if parsing file can be done better (e.g., using `eval`).
def read_tree(file_name):
    """
    Reads a tree from a file and returns the tree as a `Graph`.
    """
    file = open(file_name, "r")
    tree = Graph()
    line1 = file.readline()
    for i in range(0, len(line1)):
        if line1[i].isdigit():
            tree.add_vertex(int(line1[i]))

    line2 = file.readline()
    line2 = line2.split("}")

    edge_list = []
    for item in line2:
        item = item.replace(",", "")
        item = item.replace(" ", "")
        edge_list.append(item.replace("{", ""))

    for item in edge_list:
        if not item == "":
            index1 = item[0]
            for j in range(2, len(item)):
                tree.add_edge(int(index1), int(item[j]))

    file.close()

    return tree


def dominion_to_fnc(dominion):
    return lambda weights: dominion[weights[0]][weights[1]]


def read_hom(alpha_text):
    hom_dict = eval(alpha_text)
    return lambda node: hom_dict[int(node)]


# TODO: Pick random files in the directory instead of using a hardcoded 20.
def get_dominion_poly(tree_num):
    # Pick random dominion and homomorphism
    file_name_d = file_name_dominion(tree_num, random.randint(0, 19))
    file_name_h = file_name_homomorphism(tree_num, random.randint(0, 19))

    dominion_file = open(file_name_d, "r")
    homomorphism_file = open(file_name_h, "r")
    dominion = eval(dominion_file.read())

    dominion_fnc = dominion_to_fnc(dominion)
    tree_hom = read_hom(homomorphism_file.read())

    dom_poly = DominionPolymorphism(dominion_fnc, tree_hom)

    homomorphism_file.close()

    return dom_poly


def generate_dominions(tree, labels, tree_num, dominion_num, dominion_size):
    """
    Given a tree, generates a given number of dominions of a given size and stores them as text files. The folder
    corresponding to the tree number must already exist.
    """
    for i in range(dominion_num):
        file_name = file_name_dominion(tree_num, i)
        dominion_file = open(file_name, "w")

        dominion_file.write(str(random_dominion(dominion_size, labels, tree)))

        dominion_file.close()


def generate_homomorphisms(tree, tree_num, hom_num, n):
    """
    Given a tree, generates a given number of homomorphisms from the tree to the Hamming graph H_n, for a given n.
    """
    for i in range(0, hom_num):
        file_name = file_name_homomorphism(tree_num, i)
        hom_file = open(file_name, "w")

        hom_file.write(str(get_homomorphism(tree, n)))

        hom_file.close()


def draw_dominion(dominion, color_map, name):
    """
    Render an image from a given dominion and color map.
    Args:
        dominion (list): A list encoding a dominion.
        color_map (string): The name of a color map.
        name (string): The name of the resulting file.
    """

    plt.imsave('{}.png'.format(name), np.array(dominion),
               cmap=eval('cm.{}'.format(color_map)))
