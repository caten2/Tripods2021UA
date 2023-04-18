"""
Tests for binary image polymorphisms
"""

from binary_image_polymorphisms import RotationAutomorphism, ReflectionAutomorphism, SwappingAutomorphism,\
    BlankingEndomorphism
from mnist_training_binary import binary_train_for_zero, show

# Load some binary images from the modified MNIST training set.
training_pairs = binary_train_for_zero(100)

# Create a rotation automorphism.
rot = RotationAutomorphism()
# Choose an image to rotate.
print('Original image')
img = training_pairs[24][0]
# Display the original.
show(img)
print()
# Display the rotated image.
print('Rotated image')
show(rot[img])
print()
# We can rotate by any number of quarter turns.
print('Rotated half a turn')
rot2 = RotationAutomorphism(2)
show(rot2[img])
print()
print('Rotated three quarter turns')
rot3 = RotationAutomorphism(3)
show(rot3[img])
print()

# Create a reflection automorphism.
refl = ReflectionAutomorphism()
# Reflect our test image.
print('Reflected image')
show(refl[img])
print()
# We can compose rotations and reflections.
print('Rotated and reflected image')
show(rot[refl[img]])

# Create a swapping automorphism.
swap = SwappingAutomorphism(training_pairs[37][0])
# Display the image used for swapping.
print('Image to use for swap')
show(training_pairs[37][0])
print()
# Swap an image.
print('Swapped image')
show(swap[img])
print()

# Create a blanking endomorphism.
blank = BlankingEndomorphism(training_pairs[37][0])
# Display the image used for blanking.
print('Image to use for blanking')
show(training_pairs[37][0])
print()
# Swap an image.
print('Blanked image')
show(blank[img])
print()
