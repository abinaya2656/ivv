import matplotlib.pyplot as plt
import cv2
import numpy as np
from operator import add
from functools import reduce

# Read the image
img = cv2.imread("img.jpg")

# Function to split image into 4 parts
def split4(image):
    half_split = np.array_split(image, 2)
    res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
    return reduce(add, res)

# Split the image
split_img = split4(img)

# Display shape of split images
print(split_img[0].shape, split_img[1].shape)

# Display the four parts
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(split_img[0])
axs[0, 1].imshow(split_img[1])
axs[1, 0].imshow(split_img[2])
axs[1, 1].imshow(split_img[3])

# Function to concatenate the 4 image parts
def concatenate4(north_west, north_east, south_west, south_east):
    top = np.concatenate((north_west, north_east), axis=1)
    bottom = np.concatenate((south_west, south_east), axis=1)
    return np.concatenate((top, bottom), axis=0)

# Reconstruct the full image
full_img = concatenate4(split_img[0], split_img[1], split_img[2], split_img[3])

# Display the reconstructed image
plt.figure()
plt.imshow(full_img)
plt.show()
