"""
Relations test
"""
from relations import Relation

print('Create a binary relation on the set {0,1,2} whose members are the pairs (0,0), (0,1), and (2,0).\n\
Note that the pair (0,0) is repeated at the end of our list of pairs. Such duplicates are ignored by the constructor\n\
for the `Relation` class.')
print()
R = Relation([[0, 0], [0, 1], [2, 0], [0, 0]], 3)

print('We can display some basic information about the relation.')
print(R)
print()

print('The relation has a frozenset of tuples.')
print(R.tuples)
print()

print('In many ways it acts like a frozenset. It has a length, which is the number of tuples in the relation.')
print(len(R))
print()

print('There is a convenience function for printing the members of `R.tuples`.')
R.show()
print()

print('We can create another relation which has the same tuples and universe.\n\
Note that we can pass in any iterable of iterables as long as the innermost iterables are pairs of integers')
S = Relation([(2, 0), (0, 1), [0, 0]], 3)
print()

print('Show the basic information about both relations we created.')
print(R)
print(S)
print()

print('We can check the tuples in each of these relations.')
print('The tuples in `R`:')
R.show()
print('The tuples in `S`:')
S.show()
print()

print('Observe that `R` and `S` are different objects as far as Python is concerned.')
print(R is S)
print()

print('You can also see this by printing their object ids.')
print(id(R))
print(id(S))
print()

print('However, `R` and `S` are equal to each other.')
print(R == S)
print()

print('We can also do comparisons. We have that `R` is contained in `S`, but this containment isn\'t proper.')
print(R <= S)
print(R < S)
print()

print('If we create a relation whose set of tuples is contained in those for `R` but whose universe is a different\n\
size, we will see that these comparisons must be between relations with the same universe and arity.')
T = Relation({(0, 0), (0, 1)}, 2)
print(T.tuples < R.tuples)
try:
    print(T < R)
except AssertionError:
    print('This will be printed because an AssertionError is thrown when comparing two relations on different \
universes.')
print()

print('Naturally, comparisons in the reverse direction work, as well.')
print(R >= S)
print(R > S)
print()

print('Since `Relation` objects are hashable, we can use them as entries in tuples, members of sets, or keys of \
dictionaries.')
tup = (R, S, T)
set_of_relations = {R, S, T}
D = {R: 1, S: 'a', T: R}
print()

print('Arithmetic operations are also possible for relations.')
print('We can create the bitwise complement of a relation.')
U = ~R
U.show()
print()

print('We can take the symmetric difference of two relations if they have the same universe and arity.')
X = Relation({(0, 0, 1), (0, 1, 1)}, 2)
Y = Relation({(0, 0, 1), (1, 0, 1)}, 2)
(X ^ Y).show()
print()

print('Similarly to the order comparisons, we will get an AssertionError if we try to add two relations with different \
universes or arities.')
Z = Relation({(0, 0, 1), (0, 1, 1)}, 3)
try:
    X ^ Z
except AssertionError:
    print('This will print since we have raised an AssertionError by trying to take the symmetric difference of two\
relations with different universes.')
print()

print('Taking the set difference of two relations can be done as follows.')
(X - Y).show()
print()

print('Taking the set intersection is done using the & operator. It is bitwise multiplication.')
(X & Y).show()
print()

print('Taking the set union is done using the | operator.')
(X | Y).show()
print()

X -= Y
print(X)
