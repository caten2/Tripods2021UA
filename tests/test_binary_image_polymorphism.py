"""
Tests for binary image polymorphisms
"""
# test for pull request !

from pathlib import Path

path = Path(__file__).parent.parent.absolute() / "src"
import sys
sys.path.insert(0, path)
from binary_image_polymorphisms import RotationAutomorphism, ReflectionAutomorphism, SwappingAutomorphism,\
    BlankingEndomorphism, IndicatorPolymorphism
from mnist_training_binary import binary_train_for_zero, show

# Load some binary images from the modified MNIST training set.
training_pairs = binary_train_for_zero(100)

# Create a rotation automorphism.
rot = RotationAutomorphism()
# Choose an image to rotate.
print('Original image')
img = training_pairs[24][0]['x']
# Display the original.
show(img)
# Display the rotated image.
print('Rotated image')
show(rot[img, ])
# We can rotate by any number of quarter turns.
print('Rotated half a turn')
rot2 = RotationAutomorphism(2)
show(rot2[img, ])
print('Rotated three quarter turns')
rot3 = RotationAutomorphism(3)
show(rot3[img, ])

# Create a reflection automorphism.
refl = ReflectionAutomorphism()
# Reflect our test image.
print('Reflected image')
show(refl[img, ])
# We can compose rotations and reflections.
print('Rotated and reflected image')
show(rot[refl[img, ], ])

# Create a swapping automorphism.
swap = SwappingAutomorphism(training_pairs[37][0]['x'])
# Display the image used for swapping.
print('Image to use for swap')
show(training_pairs[37][0]['x'])
# Swap an image.
print('Swapped image')
show(swap[img, ])

# Create a blanking endomorphism.
blank = BlankingEndomorphism(training_pairs[37][0]['x'])
# Display the image used for blanking.
print('Image to use for blanking')
show(training_pairs[37][0]['x'])
# Swap an image.
print('Blanked image')
show(blank[img, ])

# Create a binary indicator polymorphism.
ind_pol = IndicatorPolymorphism(0, 0, [training_pairs[2][0]['x'], training_pairs[51][0]['x']])
# Display the images used for dot products.
print('First image for dot product')
show(training_pairs[2][0]['x'])
print('Second image used for dot product')
show(training_pairs[51][0]['x'])
# Display a pair of images to which to apply the polymorphism.
img1 = training_pairs[3][0]['x']
img2 = training_pairs[5][0]['x']
print('First input image')
show(img1)
print('Second input image')
show(img2)
# Apply the polymorphism.
print('Image obtained from polymorphism')
show(ind_pol[img1, img2])
# Change one of the inputs and check the new output.
print('New first input')
img3 = training_pairs[34][0]['x']
show(img3)
print('New image obtained from polymorphism')
show(ind_pol[img3, img2])
