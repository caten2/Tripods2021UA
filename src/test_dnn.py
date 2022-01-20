from graphviz import Graph
from dnn import Op, Neuron, NeuralNet
from typing import TypeVar, Generic

arity = 5

architecture = [
    [[0,1,2], [1,3,4], [1], [0,2], [3,4]],
    [[0,3], [0,1,2,3], [1,2,4], [2], [4]],
    [[1,2], [2], [0,1,2,4], [3,4]],
    [[1,3], [0,2], [0,1,2,3]],
    [[0,1,2]]
]

ops = [
    Op(1, lambda x: x, name='π'),
    Op(1, lambda x,y: x+y, name='x+y'),
    Op(1, lambda x,y,z: x-y*z, name='x-y*z'),
    Op(1, lambda x,y,z,w: x*y-z*w, name='x*y-z*w')
    ]

nbhd_func = lambda f: []
loss_func = lambda g,h: 0.0

dnn = NeuralNet(arity, architecture, nbhd_func, loss_func)

for layer in dnn.layers:
    for neuron in layer:
        act_func = ops[len(neuron.in_edges)-1]
        neuron.set_op(act_func)

dnn.feed_forward([9,1,5,2,3])

dot = dnn.to_graphviz('test graph', show_vals=True)
dot.render(directory='output', view=True)
