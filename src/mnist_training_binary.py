"""
Modified MNIST training set for binary image classification

Based on the MNIST training set found at http://yann.lecun.com/exdb/mnist/
"""

import numpy as np
# Load the training and test data from keras.
data = np.load('src/data/mnistAttempt.npz')
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']


def grayscale_to_binary(image, cutoff=127):
    """
    Convert a grayscale image from the MNIST training set to a binary image.

    Arguments:
        image (numpy.ndarray): A 28 by 28 numpy array containing grayscale values from 0 to 255.
        cutoff (int): Any entry in `image` which is over this value will be taken to be a 1 (black pixel). Other pixels
            will be white.

    Returns:
        list of (list of int):  A list of 28 lists, each containing 28 entries which are either 0 (white pixel) or
            1 (black pixel).
    """

    binary_image = []
    for i in range(28):
        binary_image.append([])
        for j in range(28):
            binary_image[i].append(int(image[i][j] > cutoff))
    return binary_image


def mnist_training_pairs(quantity, cutoff=127):
    """
    Construct a list of training pairs from the MNIST dataset.

    Arguments:
        quantity (int): The number of training pairs to produce. Can be at most 60000.
        cutoff (int): Any entry in `image` which is over this value will be taken to be a 1 (black pixel). Other pixels
            will be white.

    Returns:
        list: A list of pairs (x,y) where x is a binary image of a handwritten digit and y is that digit.
    """

    return [(grayscale_to_binary(x_train[i], cutoff), y_train[i]) for i in range(quantity)]


def mnist_test_pairs(quantity, cutoff=127):
    """
    Construct a list of test pairs from the MNIST dataset.

    Arguments:
        quantity (int): The number of test pairs to produce. Can be at most 10000.
        cutoff (int): Any entry in `image` which is over this value will be taken to be a 1 (black pixel). Other pixels
            will be white.

    Returns:
        list: A list of pairs (x,y) where x is a binary image of a handwritten digit and y is that digit.
    """

    return [(grayscale_to_binary(x_test[i], cutoff), y_test[i]) for i in range(quantity)]


def binary_train_for_zero(quantity, cutoff=127):
    """
    Construct a lost of pairs for training a discrete neural net to recognize handwritten images of the digit 0.

    Args:
        quantity (int): The number of test pairs to produce.
        cutoff (int): Any entry in `image` which is over this value will be taken to be a 1 (black pixel). Other pixels
            will be white.

    Returns:
        list: A list of pairs (x,y) where x is a binary image of a handwritten digit and y is either a binary image
            which is all white (if x is not a handwritten 0) or all black (if x is a handwritten 0).
    """

    all_white = 28*[28*[0]]
    all_black = 28*[28*[1]]
    old_pairs = mnist_training_pairs(quantity, cutoff=cutoff)
    new_pairs = []
    for pair in old_pairs:
        if pair[1] == 0:
            new_pairs.append(({'x': pair[0]}, (all_black,)))
        else:
            new_pairs.append(({'x': pair[0]}, (all_white,)))
    return new_pairs


def show(image):
    """
    Examine a binary image by printing its entries.
    
    Argument:
        image (list of (list of int)): The binary image to be displayed.
    """

    for row in image:
        print(''.join(str(x) for x in row))
    print()