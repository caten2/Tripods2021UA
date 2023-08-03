"""
Tests for binary image polymorphisms
"""
# test for pull request !

from pathlib import Path

path = str(Path(__file__).parent.parent.absolute() / "src")
import sys
sys.path.insert(0, path)

from test import SwappingAutomorphism, BlankingEndomorphism, IndicatorPolymorphism, HyperoctohedralAutomorphism, Relation
from mnist_training_binary import binary_train_for_zero, show

# Load some binary images from the modified MNIST training set.
training_pairs = binary_train_for_zero(100)

# Create a rotation automorphism.
randomPermutation = HyperoctohedralAutomorphism()
# Choose an image to rotate.
print('Original image')
img = Relation(training_pairs[24][0]['x'], True, 28, 2)
# Convert to relation

# Display the original.
#show(img)
# Display the rotated image.
print('Applied random permutation')
#show(randomPermutation(img))
print(randomPermutation(img, ), "hello")

# Create a swapping automorphism.
imgToCompare = Relation(training_pairs[37][0]['x'], True, 28, 2)
swap = SwappingAutomorphism(imgToCompare)
# Display the image used for swapping.
print('Image to use for swap')
show(training_pairs[37][0]['x'])
# Swap an image.
print('Swapped image')
print(swap(img, ))

# Create a blanking endomorphism.
blank = BlankingEndomorphism(imgToCompare)
# Display the image used for blanking.
print('Image to use for blanking')
show(training_pairs[37][0]['x'])
# Swap an image.
print('Blanked image')
print(blank(img, ))

# Create a binary indicator polymorphism.
firstImageForDotProduct = Relation(training_pairs[2][0]['x'], True, 28, 2)
secondImageForDotProduct = Relation(training_pairs[51][0]['x'], True, 28, 2)
imageList = [firstImageForDotProduct, secondImageForDotProduct]
ind_pol = IndicatorPolymorphism((0, 0), imageList)
# Display the images used for dot products.
print('First image for dot product')
show(training_pairs[2][0]['x'])
print('Second image used for dot product')
show(training_pairs[51][0]['x'])
# Display a pair of images to which to apply the polymorphism.
img1 = Relation(training_pairs[3][0]['x'], True, 28, 2)
img2 = Relation(training_pairs[5][0]['x'], True, 28, 2)
print('First input image')
print(img1)
print('Second input image')
print(img2)
imgListInput = [img1, img2]
# Apply the polymorphism.
print('Image obtained from polymorphism')
print(ind_pol(imgListInput,))
# Change one of the inputs and check the new output.
print('New first input')
img3 = Relation(training_pairs[34][0]['x'], True, 28, 2)
print(img3)
imgList3 = [img3, img2]
print('New image obtained from polymorphism')
print(ind_pol(imgList3))
