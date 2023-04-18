"""
Check that MNIST training/test data is functioning
"""

import mnist_training_binary
from mnist_training_binary import show


# Create a list of 1000 training pairs.
training_pairs = mnist_training_binary.mnist_training_pairs(1000)
# Display the 59th image.
show(training_pairs[59][0])
# Display the corresponding label. Can you see the digit in the above array?
print(training_pairs[59][1])
print()

# Create a list of 1000 test pairs.
test_pairs = mnist_training_binary.mnist_test_pairs(1000)
# Display the 519th image.
show(test_pairs[519][0])
# Display the corresponding label. Can you see the digit in the above array?
print(test_pairs[519][1])
print()

# Create a list of 100 training pairs for use with a discrete neural net.
zero_training_pairs = mnist_training_binary.binary_train_for_zero(100)
# This digit 5 is labeled with an all-white image (all zeroes) to indicate it is not a handwritten 0.
show(zero_training_pairs[0][0])
show(zero_training_pairs[0][1])
print()
# This digit 0 is labelled with an all-black image (all ones) to indicate it is a handwritten 0.
show(zero_training_pairs[21][0])
show(zero_training_pairs[21][1])
