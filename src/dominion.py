"""
Dominion

Tools for creating 2-dimensional dominions
"""
import random
import matplotlib.pyplot as plt
import numpy as np
from operations import Operation
from relations import Relation

# class Graph:
#     """
#
#     """
#
#     def __init__(self, root):
#         self.root = None
#         self.vertices = []
#         self.edges = {}
#
#     def add_vertex(self, vertex_name):
#         if hasattr(self, "vertices"):
#             if vertex_name not in self.vertices:
#                 self.vertices.append(vertex_name)
#         else:
#             self.vertices = [vertex_name]
#
#     def add_edge(self, vertex1, vertex2):
#         """
#         Adds edges (vertex1, vertex2) and (vertex2, vertex1) to the graph.
#         """
#
#         self.add_vertex(vertex1)
#         self.add_vertex(vertex2)
#
#         if hasattr(self, "edges"):
#             if vertex1 in self.edges:
#                 self.edges[vertex1].add(vertex2)
#             else:
#                 self.edges.update({vertex1: {vertex2}})
#             if vertex2 in self.edges:
#                 self.edges[vertex2].add(vertex1)
#             else:
#                 self.edges.update({vertex2: {vertex1}})
#         else:
#             self.edges = {vertex1: {vertex2}, vertex2: {vertex1}}
#
#     def get_neighbors(self, vertex1):
#         return list(self.edges[vertex1])
#
#     def __str__(self):
#         vertices = ""
#         edges = ""
#         if hasattr(self, "vertices"):
#             vertices = self.vertices
#         if hasattr(self, "edges"):
#             edges = self.edges
#         return f'Vertices: {vertices}\nEdges: {edges}'
#
#     def set_root(self, vertex):
#         self.root = vertex
#
#     def make_tree_file(self, tree_num):
#         file_name = file_name_tree(tree_num)
#         tree_file = open(file_name, "w")
#
#         vertices = ""
#         edges = ""
#         if hasattr(self, "vertices"):
#             vertices = self.vertices
#         if hasattr(self, "edges"):
#             edges = self.edges
#
#         tree_file.write(f'{vertices}\n{edges}')
#
#         tree_file.close()

class DominionPolymorphism(Operation):

    def __init__(self, dominion, relabeling):
        """
        Returns a homomorphism from Ham^2 to Ham.
        Arguments:
            dominion (list of list).
            relabeling: a homomorphism from the MCG of the dominion to Ham, i.e. a map from the set of labels
            in the dominions to the set of relations that preserves adjacency.
        """

        def _func(*relations):
            # Take a pair of relations, compute the hamming weight of both relations, and map to a new relation
            # according to the given dominion.
            weights = (len(relations[0]), len(relations[1]))
            temp = dominion[weights[0]][weights[1]]
            return relabeling(temp)

        Operation.__init__(self, 2, _func, False)


def random_dominion_polymorphism(size, set_of_labels):
    constraint = random_tree(set_of_labels)
    dominion, mcg = random_dominion(size ** 2 + 1, set_of_labels, constraint)
    hom = get_homomorphism(mcg, size)
    return DominionPolymorphism(dominion, lambda x: hom[x])


# --------------- Generating Dominions --------------


# TODO: No need to check partial_row entries for candidates and rejects?
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
        (list, list): A new row which is permitted to follow `row` in a dominion with the given
        labels and constraints, a list of edges to add to the minimal constraint graph
    """

    partial_row = []
    n = len(row)

    new_edges = []

    for i in range(n):
        # To determine whether this operation is potentially adding new edges to the MCG.
        # This can only happen if the entries above the one being constructed in this iteration are identical.
        potential_new_edges = False

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
            potential_new_edges = True
            if constraint_graph is None:
                candidates = set_of_labels
            else:
                candidates = candidates.union(
                    constraint_graph.get_neighbors(list(candidates)[0]))

        # Add a random candidate.
        [random_candidate] = random.sample(list(candidates), 1)
        partial_row += [random_candidate]

        if potential_new_edges:
            new_edges += [(random_candidate, row[i])]
    return partial_row, new_edges


# def random_tree(set_of_labels):
#     """
#     Construct a random tree on a given set of labels.
#
#     Args:
#         set_of_labels (iterable): The vertex labels for the tree.
#     Returns:
#         Graph: A randomly-chosen tree on the given vertices.
#     """
#
#     labels = list(set_of_labels)
#
#     tree = Graph()
#     node = labels.pop()
#     tree.add_vertex(node)
#     tree.root = node
#
#     while len(labels) > 0:
#         node = random.choice(tree.vertices)
#         for _ in range(random.randint(1, len(labels))):
#             neighbor = labels.pop()
#             tree.add_vertex(neighbor)
#             tree.add_edge(node, neighbor)
#     return tree


# def find_root(tree):
#     for v in tree.vertices:
#         if len(tree.get_neighbors(v)) == 1:
#             return v


def random_dominion(size, set_of_labels, constraint_graph=None):
    """
    Generates a random dominion given its size, set of labels, and constraint graph.

    Args:
        size (int): The size of the dominion (size * size).
        set_of_labels (iterable): The labels used to fill up the dominion.
        constraint_graph (Graph): A graph on set_of_labels.
        This argument should be a tree to ensure that no 3-cycles arise in the minimum constraint graph (MCG).
        If the default value (the empty graph) is given, a random tree will be generated and used instead.
    Returns:
        The generated dominion along with its MCG as a tuple (dominion, MCG).
    """

    mcg = Graph()

    # We keep track of the labels in the dominion to update the MCG as we go.
    label0 = random.choice(tuple(set_of_labels))
    partial_dominion = [[label0]]

    if constraint_graph is None:
        constraint_graph = random_tree(set_of_labels)

    last_label = label0
    for _ in range(size - 1):
        new_label = random.choice(constraint_graph.get_neighbors(partial_dominion[0][-1]))
        partial_dominion[0].append(new_label)

        mcg.add_edge(new_label, last_label)
        last_label = new_label

    for _ in range(size - 1):
        next_row, new_edges = new_row(partial_dominion[-1], set_of_labels, constraint_graph)
        partial_dominion.append(next_row)

        for new_edge in new_edges:
            x, y = new_edge
            mcg.add_edge(x, y)

    return partial_dominion, mcg

def random_relation(universe_size):
    """
    Returns a random binary relation on [0, universe_size - 1].
    """
    rel = []
    for i in range(universe_size):
        for j in range(universe_size):
            if random.randint(0, 1) == 1:
                rel.append((i, j))
    return Relation(rel, universe_size, arity=2)


def random_adjacent_relation(relation):
    """
    Returns a random neighbor of a given binary relation in the Hamming graph by "switching" at most 1 tuple.
    """
    size = relation.universe_size

    # Return the original relation with probability 1/(size * size).
    if random.randint(0, size * size - 1) == 0:
        return relation

    # Otherwise, flip a random tuple.
    rand_tuple = (random.randint(0, size - 1), random.randint(0, size - 1))
    rand_rel = Relation([rand_tuple], size, relation.arity)

    return relation ^ rand_rel


# TODO: Fix
def get_homomorphism(tree, n):
    """
    Returns a random homomorphism from a tree to the Hamming graph H_n.
    """
    if tree.root is not None:
        vertex = tree.root
    else:
        vertex = find_root(tree)
    neighbors = tree.get_neighbors(vertex)
    hom = {int(vertex): random_relation(n)}
    queue = [(vertex, n) for n in neighbors]

    while queue:
        vertex, new_vertex = queue.pop()
        if vertex == new_vertex:
            continue
        neighbors = tree.get_neighbors(new_vertex)
        neighbors.remove(vertex)
        new_queue = [(new_vertex, n) for n in neighbors]

        hom.update({int(new_vertex): random_adjacent_relation(hom.get(int(vertex)))})
        queue += new_queue

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
