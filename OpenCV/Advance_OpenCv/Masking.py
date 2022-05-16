import cv2 as cv
import os
import numpy as np

"""
# Masking
"""
oldwd = os.getcwd()
os.chdir("..\Files")

dg_image = cv.imread(os.getcwd() + "/dog.jpg")
os.chdir(oldwd)

# %%
dg_mask = np.zeros(dg_image.shape[:2], dtype="uint8")

circ = cv.circle(dg_mask, (dg_image.shape[1] // 2 + 10, dg_image.shape[0] // 2 - 50), 170, 255, -1)

masked_image = cv.bitwise_and(dg_image.copy(), dg_image.copy(), mask=circ)

cv.imshow("masked", masked_image)
cv.waitKey(0)

# %%
# Blur particular mask portion of an image

mask_blur = cv.blur(dg_image, (7, 7))
out = dg_image.copy()
out[circ > 0] = mask_blur[circ > 0]
cv.imshow("mask_blur", out)

cv.waitKey(0)
