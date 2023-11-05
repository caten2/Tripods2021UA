"""
Operations for use as neural net activation functions
"""


class Operation:
    """
    A finitary operation on a set.

    Unlike `Relation`s, the objects of the `Operation` class do not have an explicit reference to their universes. This
    is because in applications the universe is often more structured than an initial section of the natural numbers,
    so storing or type-checking this is expensive in general.

    Attributes:
        arity (int): The number of arguments the operation takes. This quantity should be at least 0. A 0-ary
            Operation takes empty tuples as arguments. See the method __getitem__ below for more information on this.
        func (function or constant): The function which is used to compute the output
        value of the Operation when applied to some
            inputs.
        cache_values (bool): Whether to store already-computed values of the Operation in memory.
        values (dict): If `cache_values` is True then this attribute will keep track of which input-output pairs have
            already been computed for this Operation so that they may be reused. This can be replaced by another object
            that can be indexed.
    """

    def __init__(self, arity, func, cache_values=True):
        """
        Create a finitary operation on a set.

        Arguments:
            arity (int): The number of arguments the operation takes. This quantity should be at least 0. A 0-ary
                Operation takes empty tuples as arguments. See the method __getitem__ below for more information on
                this.
            func (function): The function which is used to compute the output value of the Operation when applied to
                some inputs. If the arity is 0, pass a constant, not a function, here.
            cache_values (bool): Whether to store already-computed values of the Operation in memory.
        """

        self.arity = arity
        self.func = func
        self.cache_values = cache_values
        if self.cache_values:
            self.values = {}

    def __call__(self, *tup):
        """
        Compute the value of the Operation on given inputs.

        Argument:
            tup (tuple of int): The tuple of inputs to plug in to the Operation.
        """
        if self.arity == 0:
            return self.func
        if self.cache_values:
            if tup not in self.values:
                self.values[tup] = self.func(*tup)
            return self.values[tup]
        return self.func(*tup)

    def __getitem__(self, ops):
        """
        Form the generalized composite with a collection of operations. The generalized composite of an operation f of
        arity k with k-many operations g_i of arity n is an n-ary operation f[g_1,...,g_k] where we evaluate as
        (f[g_1,...,g_k])(x_1,...,x_n)=f(g_1(x_1,...,x_n),...,g_k(x_1,...,x_n)).

        Composite operations are not memoized, but if their constituent operations are memoized then the composite will
        perform the appropriate lookups when called rather than recomputing those values from scratch.

        Args:
            ops (Operation | iterable of Operation): The operations with which to form the generalized composite. This
                should have length `self.arity` and all of its entries should have the same arities.

        Returns:
            Operation: The result of composing the operations in question.
        """

        # When a single operation is being passed we turn it into a list.
        if isinstance(ops, Operation):
            ops = [ops]
        assert len(ops) == self.arity
        arities = frozenset(op.arity for op in ops)
        assert len(arities) == 1
        new_arity = tuple(arities)[0]
        # We treat the case where the resulting operation is nullary separately.
        if new_arity == 0:
            return Operation(new_arity, self(*(op() for op in ops)), cache_values=False)

        # Otherwise, we need to define the composite operation as a function.
        def composite(*tup):
            """
            Evaluate the composite operation.

            Args:
                *tup: A tuple of arguments to the composite operation. The length of this should be the arity of the

            Returns:
                object: The result of applying the generalized composite operation to the arguments.
            """

            return self(*(op(*tup) for op in ops))

        return Operation(new_arity, composite, cache_values=False)


class Identity(Operation):
    """

    """

    def __init__(self):
        Operation.__init__(self, 1, lambda *x: x[0], cache_values=False)


class Projection(Operation):
    """

    """

    def __init__(self, arity, coordinate):
        Operation.__init__(self, arity, lambda *x: x[coordinate], cache_values=False)


class Constant(Operation):
    """
    An operation whose value is `constant` for all inputs. The default arity is 0,
    in which case the correct way to evaluate is as f[()], not f[].
    """

    def __init__(self, constant, arity=0, cache_values=False):
        Operation.__init__(self, arity, constant, cache_values)
