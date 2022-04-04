"""
Dominion

Tools for creating 2-dimensional dominions
"""

from sage.all import *
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time


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
            candidates = set([row[0], row[1]])
            rejects = set([])
        elif i == n - 1:
            candidates = set([row[n - 2], row[n - 1], partial_row[n - 2]])
            rejects = set([])
        else:
            candidates = set([row[i - 1], row[i], partial_row[i - 1]]).intersection(set([row[i], row[i + 1]]))
            rejects = set([row[i - 1], row[i], row[i + 1], partial_row[i - 1]]).difference(candidates)
        # If there are 2 candidate labels then we are done, since we know the only two possible choices for
        # the new label. If there is only 1 candidate then either there are some rejects, in which case it must be
        # chosen, or there are no rejects, in which case a new label may appear. This new label is subject to the
        # adjacency relation for `constraint_graph`.
        if len(candidates) == 1 and len(rejects) == 0:
            if constraint_graph == None:
                candidates = set_of_labels
            else:
                candidates = candidates.union(constraint_graph.neighbors(list(candidates)[0]))
        partial_row += (random.sample(candidates, 1))
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
    G = Graph([[], []], format='vertices_and_edges')
    while len(labels) > 0:
        if G.vertices() == []:
            node = labels.pop()
            G.add_vertex(node)
        else:
            node = random.choice(list(G.vertices()))
            for _ in range(randint(1, len(labels))):
                neighbor = labels.pop()
                G.add_vertex(neighbor)
                G.add_edge(node, neighbor)
    return G


def random_dominion(size, set_of_labels, constraint_graph=None):
    """
    
    """

    if constraint_graph is None:
        partial_dominion = [random.choices(tuple(set_of_labels), k=size)]
    else:
        partial_dominion = [[random.choice(tuple(set_of_labels))]]
        for _ in range(1, size):
            partial_dominion[0].append(random.choice(constraint_graph.neighbors(partial_dominion[0][-1])))
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


