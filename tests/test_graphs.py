from graphs import Graph, load_graph_from_file, create_random_tree

G = Graph(('a', 'b', 'c'), (('a', 'b'), ('c', 'b')))
print(tuple(G.neighbors('b')))
print(G)

# G.write_to_file('graphs')

T = create_random_tree(range(10))
print(T)

print(load_graph_from_file('graphs', 0))
