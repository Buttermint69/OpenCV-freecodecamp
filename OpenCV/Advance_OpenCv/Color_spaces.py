import cv2 as cv
import os
import matplotlib.pyplot as plt

"""
# Color Spaces
"""
oldwd = os.getcwd()
os.chdir("..\Files")

dg_image = cv.imread(os.getcwd() + "/dog.jpg")
os.chdir(oldwd)
# %%
# Switching between color spaces
# BGR to Grayscale
gray = cv.cvtColor(dg_image, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)

# BGR to HSV
hsv = cv.cvtColor(dg_image, cv.COLOR_BGR2HSV)
cv.imshow("hsv", hsv)

# BGR to LAB
lab = cv.cvtColor(dg_image, cv.COLOR_BGR2LAB)
cv.imshow("lab", lab)

# BGR to RGB
rgb = cv.cvtColor(dg_image, cv.COLOR_BGR2RGB)
cv.imshow("rgb", rgb)
plt.imshow(rgb)

# or by array slicing
# rgb_asc = dg_image[:,:,::-1]
# plt.imshow(rgb_asc)

# HSV to BGR
hsv2bgr = cv.cvtColor(hsv, cv.COLOR_HLS2BGR)
cv.imshow("hsv2bgr", hsv2bgr)

plt.show()

cv.waitKey(0)
