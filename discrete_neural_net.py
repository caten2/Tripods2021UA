"""
Discrete neural net
"""

import numpy as np
import random as rand

class Operation:

    def __init__(self, order, arity, func, cache_values=True):
        self.order = order
        self.arity = arity
        self.func = func
        self.cache_values = cache_values
        if self.cache_values:
            self.values = -np.ones(arity * [order])

    def __getitem__(self, index):
        if self.cache_values and self.arity == 0:
            if self.values == -1:
                self.values = self.func(index)
        if self.cache_values and self.arity > 0:
            if self.values[index] == -1:
                self.values[index] = self.func(index)
            return int(self.values[index])
        return self.func(index)

class RandomOperation(Operation):
    
    def __init__(self, order, arity):
        Operation.__init__(self, order, arity, lambda x: rand.randint(0, self.order-1))

class ModularAddition(Operation):

    def __init__(self, order, cache_values=False):
        Operation.__init__(self, order, 2, lambda x: (x[0] + x[1]) % order, cache_values)


class ModularMultiplication(Operation):

    def __init__(self, order, cache_values=False):
        Operation.__init__(self, order, 2, lambda x: (x[0] * x[1]) % order, cache_values)


class ModularNegation(Operation):

    def __init__(self, order, cache_values=False):
        Operation.__init__(self, order, 1, lambda x: (-x) % order, cache_values)


class Constant(Operation):
    """
    An operation whose value is `constant` for all inputs. The default arity is 0,
    in which case the correct way to evaluate is as f[()], not f[].
    """

    def __init__(self, constant, order, arity=0, cache_values=False):
        Operation.__init__(self, order, arity, lambda x: constant, cache_values)

class Identity(Operation):
    
    def __init__(self,order):
        Operation.__init__(self, order, 1, lambda x: x[0], cache_values=False)

class Node:
    """
    A neuron in a neural net.
    """

    def __init__(self,activation_func,inputs):
        self.activation_func = activation_func
        self.inputs = inputs

class Layer:
    """
    
    """

    def __init__(self,nodes):
        self.nodes = nodes

class RandomNode(Node):
    
    def __init__(self,basic_ops,previous_layer):
        activation_func = rand.choice(basic_ops[rand.choice(tuple(basic_ops.keys()))])
        Node.__init__(self, activation_func, [rand.choice(previous_layer.nodes) for _ in range(activation_func.arity)])

class RandomLayer(Layer):
    
    def __init__(self,basic_ops,previous_layer,size):
        Layer.__init__(self,[RandomNode(basic_ops,previous_layer) for _ in range(size)])

class NeuralNet:
    """
    A (discrete) neural net.
    
    Attributes:
        architecture (list of Layer): The layers of the neural net, starting with the input layer,
            whose nodes should be a list of distinct variable names. Later layers should consist
            of Nodes carrying activation functions.
        basic_ops (dict of int: list): The allowable activation functions, indexed by arity.
    """
    def __init__(self,architecture,basic_ops):
        self.architecture = architecture
        self.basic_ops = basic_ops
        # self.signature = {}

    def feed_forward(self,x):
        """
        Feed the values `x` forward through the neural net.
        
        Args:
            x (dict of str: int): An assignment of variable names to values.
        """
        
        current_vals = x
        for layer in self.architecture[1:]:
            for node in layer.nodes:
                index = tuple(current_vals[input_node] for input_node in node.inputs)
                current_vals[node] = node.activation_func[index]
        return tuple(current_vals[node] for node in self.architecture[-1].nodes)
    
    def random_train(self,z_tup,loss_func):
        """
        Args:
            z_tup (iterable): Training pairs (x,y) where x is a dictionary of inputs and y is a tuple of outputs.
            loss_func (function): The loss function to use for training.
        """

        layer = rand.choice(self.architecture[1:])
        node = rand.choice(layer.nodes)
        operations = []
        emp_loss = []
        for op in self.basic_ops[node.activation_func.arity]:
            vals = []
            for (x,y) in z_tup:
                node.activation_func = op
                vals.append(loss_func(self.feed_forward(x),y))
            operations.append(op)
            emp_loss.append(np.average(vals))
        node.activation_func = operations[emp_loss.index(min(emp_loss))]

    def operation_tweak_train(self,z_tup,loss_func):
        """
        Args:
            z_tup (iterable): Training pairs (x,y) where x is a dictionary of inputs and y is a tuple of outputs.
            loss_func (function): The loss function to use for training.
        """

        op = rand.choice(self.basic_ops[rand.choice(list(self.basic_ops.keys()))])
        while type(op) is Identity:
            op = rand.choice(self.basic_ops[rand.choice(list(self.basic_ops.keys()))])
        index = tuple(rand.randint(0,op.order-1) for _ in range(op.arity))
        emp_loss = []
        for k in range(op.order):
            op.values[index] = k
            vals = []
            for (x,y) in z_tup:
                vals.append(loss_func(self.feed_forward(x),y))
            emp_loss.append(np.average(vals))
        print(emp_loss)
        op.values[index] = emp_loss.index(min(emp_loss))
        print(op.values[index])

    def activation_tweak_train(self,z_tup,loss_func):
        """
        Args:
            z_tup (iterable): Training pairs (x,y) where x is a dictionary of inputs and y is a tuple of outputs.
            loss_func (function): The loss function to use for training.
        """

        layer = rand.choice(self.architecture[1:])
        node = rand.choice(layer.nodes)
        while type(node.activation_func) is Identity:
            layer = rand.choice(self.architecture[1:])
            node = rand.choice(layer.nodes)
        op = node.activation_func
        index = tuple(rand.randint(0,op.order-1) for _ in range(op.arity))
        emp_loss = []
        for k in range(op.order):
            new_op = RandomOperation(op.order,op.arity)
            new_op.values = op.values[:]
            new_op.values[index] = k
            node.activation_func = new_op
            vals = []
            for (x,y) in z_tup:
                vals.append(loss_func(self.feed_forward(x),y))
            emp_loss.append(np.average(vals))
        print(emp_loss)
        new_op = RandomOperation(op.order,op.arity)
        new_op.values = op.values[:]
        new_op.values[index] = emp_loss.index(min(emp_loss))
        node.activation_func = new_op
        print(op.values[index])

    def faster_activation_tweak_train(self,z_tup,loss_func):
        """
        Args:
            z_tup (iterable): Training pairs (x,y) where x is a dictionary of inputs and y is a tuple of outputs.
            loss_func (function): The loss function to use for training.
        """

        layer = rand.choice(self.architecture[1:])
        node = rand.choice(layer.nodes)
        while type(node.activation_func) is Identity:
            layer = rand.choice(self.architecture[1:])
            node = rand.choice(layer.nodes)
        op = node.activation_func
        index = tuple(rand.randint(0,op.order-1) for _ in range(op.arity))
        emp_loss = []
        new_val = rand.randint(0,op.order-1)
        for k in [op.values[index],new_val]:
            new_op = RandomOperation(op.order,op.arity)
            new_op.values = op.values[:]
            new_op.values[index] = k
            node.activation_func = new_op
            vals = []
            for (x,y) in z_tup:
                vals.append(loss_func(self.feed_forward(x),y))
            emp_loss.append(np.average(vals))
        print(emp_loss)
        new_op = RandomOperation(op.order,op.arity)
        new_op.values = op.values[:]
        if emp_loss[1] < emp_loss[0]:
            new_op.values[index] = new_val
        node.activation_func = new_op
        print(op.values[index])

class RandomNeuralNet(NeuralNet):
    
    def __init__(self,order,inputs,outputs,depth,breadth,signature):
        """
        Args:
            order (int): The number of elements of the universe.
            inputs (iterable of str): The names of the input nodes.
            outputs (int): The number of output nodes.
            depth (int): The number of layers in the neural net.
            breadth (int): The maximum number of nodes in a layer.
            signature (dict of int: int): Maps an arity to the (nonzero) number of basic operations of that arity.
        """
        
        basic_ops = {n: [RandomOperation(order,n) for _ in range(signature[n])] for n in signature.keys()}
        nonidentity_basic_ops = basic_ops.copy()
        if 0 in basic_ops.keys():
            basic_ops[1].append(Identity(order))
        else:
            basic_ops[1] = [Identity(order)]
        architecture = [Layer(inputs)]
        for _ in range(depth-2):
            architecture.append(RandomLayer(basic_ops,architecture[-1],rand.randint(1,breadth)))
        architecture.append(RandomLayer(nonidentity_basic_ops,architecture[-1],outputs))
        NeuralNet.__init__(self,architecture,basic_ops)


