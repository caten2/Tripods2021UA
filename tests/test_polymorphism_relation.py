"""
Tests for binary image polymorphisms
"""
from pathlib import Path

path = str(Path(__file__).parent.parent.absolute() / "src")
import sys
sys.path.insert(0, path)

from test import SwappingAutomorphism, BlankingEndomorphism, IndicatorPolymorphism, HyperoctohedralAutomorphism, Relation, swapping
from mnist_training_binary import binary_train_for_zero, show

# In order to save space and reduce runtime, we will store images as relations 
# instead of lists of lists of integers. Instead of storing the background pixels
# as 0s and the pixels containing the writing as 1s in a matrix represented by a 
# list of lists, we will save the indices (0-indexed) of the 1s in the matrix,
# and perform morphisms on only the list of those indices. The indices not included
# in this list will represent the 0s. 

# Structure-wise, using the same language to describe them as we would in category theory
# these lists of indices will belong to a class called Relation.
# Either a list of lists of integers or a list of tuples of integers can be passed into
# this class, and both will be stored as a relation, ie a list of indices (tuples).

# The polymorphisms listed in this file have a clear visual representation. Thus, in
# order to see how the polymorphisms are changing the images, we need to make a function
# to make an image (list of lists) from a binary relation. 

def relationToImage(x):
    length = x.n
    image = []
    for i in range(x.n):
        row_list = []
        for j in range(x.n):
            if (i, j) in x.rList:
                row_list.append(1)
            else:
                row_list.append(0)
        image.append(row_list)
            
    return image

# Now, to start testing polymorphisms.

# Load some binary images from the modified MNIST training set.
training_pairs = binary_train_for_zero(100)

# Create a permutation automorphism.
randomPermutation = HyperoctohedralAutomorphism()
# Choose an image to rotate, and convert it to a Relation.
print('Original image')
img = Relation(training_pairs[24][0]['x'], True, 28, 2)
# Display the original.
show(relationToImage(img))
# Display the rotated image.
print('Applied random permutation')
show(relationToImage(randomPermutation(img, )))


# Create a swapping automorphism.
print('Swapping time!')
imgToCompare = Relation(training_pairs[37][0]['x'], True, 28, 2)
swap = SwappingAutomorphism(imgToCompare)
# Display the image used for swapping.
print('Image to use for swap')
show(relationToImage(imgToCompare))
# Swap an image.
print('Swapped image')
show(relationToImage(swap(img, )))


# Create a blanking endomorphism.
print('Blanking time!')
blank = BlankingEndomorphism(imgToCompare)
# Display the image used for blanking.
print('Image to use for blanking')
show(relationToImage(imgToCompare))
# Blank an image.
print('Blanked image')
show(relationToImage(blank(img, )))



# Create a binary indicator polymorphism.
print('Indicator time!')
firstImageForDotProduct = Relation(training_pairs[2][0]['x'], True, 28, 2)
secondImageForDotProduct = Relation(training_pairs[51][0]['x'], True, 28, 2)
imageList = [firstImageForDotProduct, secondImageForDotProduct]
ind_pol = IndicatorPolymorphism((0, 0), imageList)
# Display the images used for dot products.
print('First image for dot product')
show(relationToImage(firstImageForDotProduct))
print('Second image used for dot product')
show(relationToImage(secondImageForDotProduct))
# Display a pair of images to which to apply the polymorphism.
img1 = Relation(training_pairs[3][0]['x'], True, 28, 2)
img2 = Relation(training_pairs[5][0]['x'], True, 28, 2)
print('First input image')
show(relationToImage(img1))
print('Second input image')
show(relationToImage(img2))
imgListInput = [img1, img2]
# Apply the polymorphism.
print('Image obtained from polymorphism')
show(relationToImage(ind_pol(imgListInput,)))
# Change one of the inputs and check the new output.
print('New first input')
img3 = Relation(training_pairs[34][0]['x'], True, 28, 2)
show(relationToImage(img3))
imgList3 = [img3, img2]
print('New image obtained from polymorphism')
show(relationToImage(ind_pol(imgList3)))

