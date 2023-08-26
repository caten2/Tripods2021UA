"""
Modified MNIST training set for binary image classification
"""
import json
from relations import Relation
from itertools import product


def import_mnist_data(data_type):
    """
    Create an iterator for MNIST data. The resulting JSON files have each line representing a greyscale image of a
    handwritten digit. Each line is a dictionary whose keys are integers between 1 and 255, or the string 'label'.
    The values associated to the integer keys are lists of the coordinates at which the greyscale value for the MNIST
    image is equal to the key. For example, if the greyscale value 25 is found at coordinate [2,14], then the key 25
    would be associated to a list of pairs, one of which is [2,14]. Any pairs not belonging to a value in the dictionary
    are assumed to be assigned greyscale value 0. The 'label' key has an integer value between 0 and 9, indicated the
    intended handwritten digit for the corresponding image.

    Argument:
        data_type (str): Either 'train' or 'test', depending on which data one would like to convert.

    Yields:
        dict: The dictionary of data specifying a greyscale image and its intended handwritten digit.
    """

    with open('..//mnist//{}_data.json'.format(data_type), 'r') as read_file:
        for line in read_file:
            data = json.loads(line)
            # By default, all the integer keys in the dictionary returned from the JSON file will be converted to
            # strings. Let's undo this.
            cleaned_data = {int(key): data[key] for key in data if key != 'label'}
            cleaned_data['label'] = data['label']
            yield cleaned_data


def greyscale_to_binary(image, cutoff=127):
    """
    Convert a greyscale image from the MNIST training set to a binary relation.

    Arguments:
        image (dict): A dictionary representing a greyscale image as described in `import_mnist_data`.
        cutoff (int): Any pixel coordinates in `image` which are over this value will be taken to be in the relation.

    Returns:
        Relation: A binary relation on a universe of size 28 whose pairs are those coordinates from `image` which are at
            least as large as `cutoff`.
    """

    pairs = []
    for val in range(cutoff, 256):
        if val in image:
            pairs += image[val]
    return Relation(pairs, 28)


def mnist_binary_relations(data_type, quantity, cutoff=127):
    """
    Create an iterator for binary relations coming from MNIST data.

    Arguments:
        data_type (str): Either 'train' or 'test', depending on which data one would like to examine.
        quantity: The number of training pairs to produce.
        cutoff: Any pixel coordinates in a greyscale image which are over this value will be taken to be in the
            corresponding relation.

    Yields:
        tuple: A binary relation corresponding to a greyscale image from an MNIST dataset and its corresponding integer
            label.
    """

    data = import_mnist_data(data_type)
    for _ in range(quantity):
        dic = next(data)
        yield greyscale_to_binary(dic, cutoff), dic['label']


def binary_mnist_for_zero(data_type, quantity, cutoff=127):
    """
    Construct a tuple of pairs for training or testing a discrete neural net to recognize handwritten images of the
    digit 0.

    Arguments:
        data_type (str): Either 'train' or 'test', depending on which data one would like to examine.
        quantity (int): The number of test pairs to produce.
        cutoff (int): Any pixel coordinates in a greyscale image which are over this value will be taken to be in the
            corresponding relation.

    Yields:
        tuple: A pair whose first entry is a dictionary indicating that a binary relation representing a handwritten
            digit is to be fed into a discrete neural net as the input `x` and whose second entry indicates that the
            resulting output should either be an image which is all black or all white, depending on whether the input
            was a handwritten 0 or not.
    """

    all_white = Relation(tuple(), 28, 2)
    all_black = Relation(product(range(28), repeat=2), 28)
    old_pairs = mnist_binary_relations(data_type, quantity, cutoff=cutoff)
    for pair in old_pairs:
        if pair[1] == 0:
            yield {'x': pair[0]}, (all_black,)
        else:
            yield {'x': pair[0]}, (all_white,)
