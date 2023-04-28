"""
Check gAlpha with images from the MNIST set
"""

import sys
sys.path.insert(0, r"C:\Users\2019r\vscode-workspace\395W\Tripods2021UA\src")
import mnist_training_binary
from mnist_training_binary import show
import dominion
from dominion import getGAlpha


# Create a list of 100 training pairs.
training_pairs = mnist_training_binary.mnist_training_pairs(100)


gAlpha=getGAlpha(3)
print(gAlpha.__getitem__([training_pairs[1][0], training_pairs[2][0]]))
