"""
Graphs and trees
"""
import itertools
import json
import random
from pathlib import Path


def take_other_element(p, e):
    """

    Arguments:
        p (frozenset): The pair of elements, one of which is meant to be `e`.
        e (Object): The element in question.
    """

    for x in p:
        if x != e:
            return x


class Graph:
    """
    A simple graph. That is, a set of vertices with unordered pairs of vertices as edges.

    Attributes:
        vertices (frozenset): The vertices of the graph.
        edges (frozenset of frozenset): The unordered pairs of vertices constituting the edges of the graph.
    """

    def __init__(self, vertices=frozenset(), edges=frozenset()):
        """
        Create a graph with given vertices and edges.

        Arguments:
            vertices (frozenset): The vertices of the graph.
            edges (frozenset of frozenset): The unordered pairs of vertices constituting the edges of the graph.
        """

        self.vertices = frozenset(vertices)
        self.edges = frozenset(frozenset(edge) for edge in edges)

    def neighbors(self, vertex):
        """
        Construct an iterator through the neighbors of a vertex in the graph.

        Argument:
            vertex (Object): The vertex for which we find neighbors.

        Returns:
            iterator: The neighbors of the vertex in question.
        """

        return (take_other_element(edge, vertex) for edge in self.edges if vertex in edge)

    def __repr__(self):

        return "A Graph with {} vertices and {} edges.".format(len(self.vertices), len(self.edges))

    def __str__(self):

        vertices = '{' + ', '.join(map(str, self.vertices)) + '}'
        edges = '{' + ', '.join('{' + ', '.join(map(str, edge)) + '}' for edge in self.edges) + '}'
        return "A Graph with vertex set {} and edge set {}.".format(vertices, edges)

    # This method isn't strictly needed for the current implementation but is nice for debugging.
    def write_to_file(self, filename):
        """
        Write a Graph to a json file. A file with the appropriate name will be created if it doesn't already exist.
        The Graph will be appended to the next line of the file if it already exists.

        Argument:
            filename (str): The name of the output file.
        """

        with open(str(Path(__file__).parent.resolve()) + '//..//src//output//{}.json'.format(filename),
                  'a') as write_file:
            # The Graph is rendered as a pair of lists, since frozensets are not serializable in json.
            json.dump((tuple(self.vertices), tuple(map(tuple, self.edges))), write_file)
            write_file.write('\n')


def load_graph_from_file(filename, graph_number):
    """


    Attributes:
        filename (str): The name of the json file containing the Graph.
        graph_number (int): The line number in the file describing the desired Graph.
    """

    with open(str(Path(__file__).parent.resolve()) + '//..//src//output//{}.json'.format(filename),
              'r') as read_file:
        unprocessed_graph = itertools.islice(read_file, graph_number, graph_number+1).__next__()
    return Graph(*json.loads(unprocessed_graph))


def create_random_tree(vertices):
    """

    """

    unplaced_vertices = set(vertices)
    root_vertex = unplaced_vertices.pop()
    placed_vertices = [root_vertex]
    edges = set()
    while unplaced_vertices:
        new_vertex = unplaced_vertices.pop()
        old_vertex = random.choice(placed_vertices)
        edges.add((old_vertex, new_vertex))
        placed_vertices.append(new_vertex)
    return Graph(vertices, edges)
