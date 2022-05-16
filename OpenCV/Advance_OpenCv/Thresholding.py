import cv2 as cv
import os
"""
# Thresholding
"""
oldwd = os.getcwd()
os.chdir("..\Files")

dg_image = cv.imread(os.getcwd() + "/dog.jpg")
os.chdir(oldwd)

gray_scale = cv.cvtColor(dg_image, cv.COLOR_BGR2GRAY)

# %%
# Simple Thresholding

threshold, thresh = cv.threshold(gray_scale, 125, 255, cv.THRESH_BINARY)
cv.imshow("Simple Threshold", thresh)

threshold_, thresh_inv = cv.threshold(gray_scale, 205, 255, cv.THRESH_BINARY_INV)
cv.imshow("Simple Threshold Inverse", thresh_inv)

# Adaptive Thresholding

thresh_adapt = cv.adaptiveThreshold(gray_scale, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY, 17, -9)
cv.imshow("Adaptive Threshold Inverse", thresh_adapt)

cv.waitKey(0)