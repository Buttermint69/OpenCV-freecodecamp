import cv2 as cv
import numpy as np
import os

"""
# Image Transformations
"""
oldwd = os.getcwd()
os.chdir("..\Files")

img = cv.imread(os.getcwd() + "/ape2.png")
os.chdir(oldwd)

# %%
# Translation; shifting image
def trans(image, x, y):
    """
    -x ---> left shift
    -y ---> up
     x ---> right shift
     y ---> down
    """
    trans_m = np.float32([[1, 0, x], [0, 1, y]])
    dim = (image.shape[1], image.shape[0])
    return cv.warpAffine(image, trans_m, dim)


translated = trans(img, -100, 100)


# Rotation
def rotate(image, angle, rotPoint=None):
    (height, width) = image.shape[:2]

    if rotPoint is None:
        rotPoint = (width // 2, height // 2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.)
    dimension = (width, height)

    return cv.warpAffine(image, rotMat, dimension)


rotated = rotate(img, 45)
rt = rotate(img, angle=-40, rotPoint=(50, 50))

# Flipping
flip = cv.flip(img, 1)

cv.imshow("translated", translated)
cv.imshow("rot", rotated)
cv.imshow("flip", flip)

cv.waitKey(0)
