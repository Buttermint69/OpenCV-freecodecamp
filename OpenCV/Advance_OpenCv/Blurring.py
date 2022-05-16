import cv2 as cv
import os

"""
# Blurring Techniques
"""

oldwd = os.getcwd()
os.chdir("..\Files")

dg_image = cv.imread(os.getcwd() + "/dog.jpg")
os.chdir(oldwd)
# %%
# Average Blurring over kernels
blur = cv.blur(dg_image, (7, 7))
cv.imshow("Average Blur", blur)

# Gaussian Blur
G_blur = cv.GaussianBlur(dg_image, (7, 7), 0)
cv.imshow("Gaussian Blur", G_blur)

# Median Blur
me_blur = cv.medianBlur(dg_image, 7)
cv.imshow("Median BLur", me_blur)

# Bilateral Blur
bi_blur = cv.bilateralFilter(dg_image, 10, 55, 25)
cv.imshow("Bilateral Blur", bi_blur)

cv.waitKey(0)
