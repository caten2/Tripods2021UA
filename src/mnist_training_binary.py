"""
Modified MNIST training set for binary image classification
"""
import json
from pathlib import Path
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

    with open(str(Path(__file__).parent.resolve()) + '//..//mnist//{}_data.json'.format(data_type), 'r') as read_file:
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


def mnist_binary_relations(data_type, cutoff=127):
    """
    Create an iterator for binary relations coming from MNIST data.

    Arguments:
        data_type (str): Either 'train' or 'test', depending on which data one would like to examine.
        cutoff: Any pixel coordinates in a greyscale image which are over this value will be taken to be in the
            corresponding relation.

    Yields:
        tuple: A binary relation corresponding to a greyscale image from an MNIST dataset and its corresponding integer
            label.
    """

    data = import_mnist_data(data_type)
    for dic in data:
        yield greyscale_to_binary(dic, cutoff), dic['label']


def build_training_data(pairs, data_type, cutoff=127):
    """
    Create an iterable of pairs for training or testing a discrete neural net using the MNIST datasets. Either the
    train data or the test data from MNIST may be used.

    The following values provided in `pairs` will be substituted for arbitrary entries from the MNIST data.
        0, 1, 2, 3, 4, 5, 6, 7, 8, or 9: These `int`s will be replaced with a corresponding handwritten digit from
            MNIST.
        'Empty', 'Full': These strings will be replaced with the empty binary relation and the full binary relation,
            respectively.

    Arguments:
        pairs (iterable of tuple): A sequence of pairs, the first entry being a tuple of inputs and the second entry
            being a tuple of outputs. It is assumed that all the first-entry tuples have the same length, which is the
            number of input nodes in a neural net to be trained/tested on such data. Similarly, the second-entry tuples
            are assumed to have the same length, which is the number of output nodes in a neural net to be
            trained/tested on such data. See the description above for possible values that these tuples may contain.
        data_type (str): Either 'train' or 'test', depending on which data one would like to examine.
        cutoff: Any pixel coordinates in a greyscale image which are over this value will be taken to be in the
            corresponding relation.

    Yields:
        tuple: A pair whose first entry is a dictionary indicating that a tuple of binary relations is to be fed into a
            discrete neural net as the inputs `x0`, `x1`, `x2`, etc. and whose second entry is a tuple of binary
            relations which should appear as the corresponding outputs.
    """

    # Create a dictionary for the substitutions described above. The images corresponding to the digits will be updated
    # dynamically from the MNIST training data.
    substitution_dic = {i: None for i in range(10)}
    substitution_dic['Empty'] = Relation(tuple(), 28, 2)
    substitution_dic['Full'] = Relation(product(range(28), repeat=2), 28)
    # Load the MNIST data
    data = mnist_binary_relations(data_type, cutoff)
    # Initialize the images corresponding to the digits.
    for i in range(10):
        # For each digit, we try to find a candidate image.
        while not substitution_dic[i]:
            # We pull the next image from MNIST.
            new_image = next(data)
            # If an image for that digit hasn't been found yet, regardless of whether it was the one we intended to look
            # for, that image will be added as the one representing its digit in `substitution_dic`.
            if not substitution_dic[new_image[1]]:
                substitution_dic[new_image[1]] = new_image[0]
    for pair in pairs:
        # Update one of the digits using the next values from MNIST.
        new_image = next(data)
        substitution_dic[new_image[1]] = new_image[0]
        yield {'x{}'.format(i): substitution_dic[pair[0][i]] for i in range(len(pair[0]))}, \
              tuple(substitution_dic[pair[1][i]] for i in range(len(pair[1])))


def binary_mnist_zero_one(quantity_of_zeroes, data_type, quantity_of_ones=None, cutoff=127):
    """
    Create a data set for training a discrete neural net to recognize handwritten zeroes and ones. Zeroes are labeled
    with the empty relation and ones are labeled with the full relation.

    Args:
        quantity_of_zeroes (int): The number of examples of handwritten zeroes to show.
        data_type (str): Either 'train' or 'test', depending on which data one would like to examine.
        quantity_of_ones (int): The number of examples of handwritten ones to show.
        cutoff: Any pixel coordinates in a greyscale image which are over this value will be taken to be in the
            corresponding relation.

    Returns:
        iterable: An iterable of training data where handwritten zeroes and ones are mapped to full and empty relations.
    """

    # If the number of ones to use is not specified, it is assumed to be the same as the number of zeroes.
    if not quantity_of_ones:
        quantity_of_ones = quantity_of_zeroes
    pairs = [((0,), ('Full',)) for _ in range(quantity_of_zeroes)]
    pairs += [((1,), ('Empty',)) for _ in range(quantity_of_ones)]
    return build_training_data(pairs, data_type, cutoff)
