import cv2 as cv
import os
import numpy as np
"""
# Rescaling Images and videos
"""
oldwd = os.getcwd()
os.chdir("..\Files")

img = cv.imread(os.getcwd() + "/ape2.png")
os.chdir(oldwd)


# %%
def rescale(frame, scale_factor=0.5):
    width = int(frame.shape[1] * scale_factor)
    height = int(frame.shape[0] * scale_factor)

    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)


re_img = rescale(img)
cv.imshow("nft", img)
cv.imshow("nft_resized", re_img)
cv.waitKey(0)

