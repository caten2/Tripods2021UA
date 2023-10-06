# Discrete neural nets

This repo contains code and examples of implementing the notion of a discrete neural net and polymorphic learning in
Python. For more information on these notions, see [the corresponding preprint](https://arxiv.org/abs/2308.00677) on the
arXiv.

### Git workflow

1) Fork main repo (https://github.com/caten2/Tripods2021UA). This is different than just cloning. There is a button on
   GitHub for this.
2) Clone your fork to your machine.
3) Make a branch on your machine.
4) At any time, you can push your changes to your personal fork of the repo.
5) On your machine, pull from the main repo's main branch.
6) On your machine, merge the main branch into your new branch.
7) Resolve any conflicts that arise, then merge your new branch into the main branch.
8) Push your work to your personal fork of the repo.
9) Make a pull request to pull your fork's main into the original repo's main.

### Project structure

In the lists which follow, scripts marked with (ORGANIZE) are not part of the current, functioning implementation. These
may be in the process of refactoring or being reorganized into another part of the repo.

The scripts that define basic components of the system are in the `src` folder. These are:

* `arithmetic_operations.py`: Definitions of arithmetic operations modulo some positive integer. These are used to test
the basic functionality of the `NeuralNet` class.
* `binary_image_polymorphisms.py`: Definitions of polymorphisms of the Hamming graph, as well as a neighbor function for
  the learning algorithm implemented in `neural_net.py`. (ORGANIZE)
* `neural_net.py`: Definition of the `NeuralNet` class, including feeding forward and learning.
* `dominion.py`: Tools for creating dominions, a combinatorial object used in the definition of the dominion
  polymorphisms in `polymorphisms.py`. (ORGANIZE)
* `hyperoctohedral.py`: Definitions of polymorphisms of the Hamming graph which come from the action of the
  hyperoctahedral group. (ORGANIZE)
* `mnist_training_binary.py`: Describes how to manufacture binary relations from the MNIST dataset which can be passed
  as arguments into the polymorphisms in `polymorphisms.py`.
* `operations.py`: Definitions pertaining to the `Operation` class, whose objects are to be thought of as operations in
  the sense of universal algebra/model theory.
* `polymorphisms.py`: Definitions of polymorphisms of the Hamming graph, as well as a neighbor function for
  the learning algorithm implemented in `neural_net.py`.
* `random_neural_net.py`: Tools for making `NeuralNet` objects with randomly-chosen architectures and activation
  functions.
* `relations.py`: Definitions pertaining to the `Relation` class, whose objects are relations in the sense of model
theory.
* `test.py`: A test script which should be moved to the `tests` directory. (ORGANIZE)

The scripts that run various tests and example applications of the system are in the `tests` folder. These are:

* Those in the subdirectory `binary_relation_polymorphisms`: (Add description.)
* `example_dominion.py`: (Add description.) (ORGANIZE)
* `test_binary_image_train_gAlpha.py`: (Add description.) (ORGANIZE)
* `test_binary_relation_polymorphisms`: Examples of the basic functionality for the polymorphisms defined in
`polymorphisms.py` when applied to binary relations.
* `test_dominion.py`: (Add description.) (ORGANIZE)
* `test_gAlpha.py`: (Add description.) (ORGANIZE)
* `test_mnist_training_binary.py`: Verification that MNIST training data is being loaded correctly from the training
dataset.
* `test_neural_net.py`: Examples of creating `NeuralNet`s using activation functions from
`arithmetic_operations.py` and the `RandomOperation` from `random_neural_net.py`.
* `test_polymorphism_relation.py`: (Add description.) (ORGANIZE)
* `test_relations.py`: Examples of the basic functionality for the `Relation`s defined in `relations.py`.

### Environment

Make sure that the scripts in `tests` can import from scripts in `src`. In PyCharm this is accomplished most easily by
going to Settings -> Project Structure and making sure that `src` is set as a "source" folder.