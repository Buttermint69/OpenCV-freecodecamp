import os
import cv2 as cv
import numpy as np

"""
# Contour Detection
"""

# %%

oldwd = os.getcwd()
os.chdir("..\Files")

dg_image = cv.imread(os.getcwd() + "/dog.jpg")
os.chdir(oldwd)

cv.imshow("dog", dg_image)

blank = np.zeros(dg_image.shape, dtype="uint8")

gray = cv.cvtColor(dg_image, cv.COLOR_BGR2GRAY)
# cv.imshow("", gray)

blur_dog = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
# cv.imshow("blur", blur_dog)

canny_d = cv.Canny(blur_dog, 125, 175)
cv.imshow("canny_dog", canny_d)

ret, thresh = cv.threshold(blur_dog, 125, 255, cv.THRESH_BINARY)
cv.imshow("thresh", thresh)

contours, hierarchies = cv.findContours(canny_d, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

print(f"{len(contours)} contours found")

cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow("drawn", blank)

cv.waitKey(0)
