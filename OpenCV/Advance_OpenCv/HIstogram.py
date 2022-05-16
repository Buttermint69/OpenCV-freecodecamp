import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
"""
# Histogram
## Allows to visualize the pixel intensity distribution in an image
"""
oldwd = os.getcwd()
os.chdir("..\Files")

dg_image = cv.imread(os.getcwd() + "/dog.jpg")
os.chdir(oldwd)
# %%
gray_scale = cv.cvtColor(dg_image, cv.COLOR_BGR2GRAY)
# cv.imshow("gray_scale", gray_scale)

dg_mask = np.zeros(dg_image.shape[:2], dtype="uint8")
circ = cv.circle(dg_mask, (dg_image.shape[1] // 2 + 10, dg_image.shape[0] // 2 - 50), 170, 255, -1)

masked_image = cv.bitwise_and(gray_scale.copy(), gray_scale.copy(), mask=circ)
cv.imshow("gray_scale_mask", masked_image)

# Grayscale histogram
gray_hist = cv.calcHist([gray_scale], [0], masked_image, [256], [0, 256])
plt.figure()

plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel(" # of pixels")
plt.plot(gray_hist)
plt.xlim([0, 256])

plt.show()

cv.waitKey(0)

# %%
# Color Histogram
cv.imshow("image", dg_image)
plt.title("Color Histogram")
plt.xlabel("Bins")
plt.ylabel(" # of pixels")
plt.xlim([0, 256])

col = ("b", "g", "r")
for i, c in enumerate(col):
    color_hist = cv.calcHist([dg_image], [i], None, [256], [0, 256])
    plt.plot(color_hist, color=c)

plt.show()
cv.waitKey(0)
