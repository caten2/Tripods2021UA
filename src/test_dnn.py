from graphviz import Graph
from dnn import Op, Neuron, NeuralNet
from typing import TypeVar, Generic

arity = 5

architecture = [
    [[0,1,2], [1,3,4], [1], [0,2], [3,4]],
    [[0,3], [0,1,2,3], [1,2,4], [2], [4]],
    [[1,2], [2], [0,1,2,4], [3,4]],
    [[0,1,2,3]]
]

ops = [
    Op(1, lambda x: x, name='id'),
    Op(1, lambda x,y: x, name='id'),
    Op(1, lambda x,y,z: x, name='id'),
    Op(1, lambda x,y,z,w: x, name='id')
    ]

nbhd_func = lambda f: []
loss_func = lambda g,h: 0.0

dnn = NeuralNet(arity, architecture, nbhd_func, loss_func)

for layer in dnn.layers:
    for neuron in layer:
        act_func = ops[len(neuron.in_edges)-1]
        neuron.set_op(act_func)

dnn.feed_forward([0,0,0,0,0])

dot = dnn.to_graphviz('test graph')
dot.render(directory='output', view=True)
