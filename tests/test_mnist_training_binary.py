"""
Check that MNIST training/test data is functioning
"""
import mnist_training_binary

# Create a list of 1000 training pairs.
training_pairs = tuple(mnist_training_binary.mnist_binary_relations('train', 1000))
# Display the 59th image.
training_pairs[59][0].show('sparse')
# Display the corresponding label. Can you see the digit in the above array?
print(training_pairs[59][1])
print()

# Create a list of 1000 test pairs.
test_pairs = tuple(mnist_training_binary.mnist_binary_relations('test', 1000))
# Display the 519th image.
test_pairs[519][0].show('sparse')
# Display the corresponding label. Can you see the digit in the above array?
print(test_pairs[519][1])
print()

# Create a list of 100 training pairs for use with a discrete neural net.
zero_training_pairs = tuple(mnist_training_binary.binary_mnist_for_zero('train', 100))
# This digit 5 is labeled with an all-white image (all zeroes) to indicate it is not a handwritten 0.
zero_training_pairs[0][0]['x'].show('sparse')
zero_training_pairs[0][1][0].show('binary_pixels')
print()
# This digit 0 is labelled with an all-black image (all ones) to indicate it is a handwritten 0.
zero_training_pairs[21][0]['x'].show('sparse')
zero_training_pairs[21][1][0].show('binary_pixels')
