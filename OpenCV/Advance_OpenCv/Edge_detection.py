import cv2 as cv
import os
import numpy as np

"""
# Edge Detection

"""
oldwd = os.getcwd()
os.chdir("..\Files")

dg_image = cv.imread(os.getcwd() + "/dog.jpg")
os.chdir(oldwd)

gray = cv.cvtColor(dg_image, cv.COLOR_BGR2GRAY)

# %%
# Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow("Laplacian edge detection", lap)

# Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
sobelxy = cv.bitwise_or(sobelx, sobely)

# cv.imshow("Sobel X", sobelx)
# cv.imshow("Sobel Y", sobely)
cv.imshow("Sobel XY_combined", sobelxy)

# Canny
can = cv.Canny(gray, 100, 150)
cv.imshow("canny", can)
cv.waitKey(0)
