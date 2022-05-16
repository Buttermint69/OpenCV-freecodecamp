import cv2 as cv
import os

"""
# Basic function
"""
oldwd = os.getcwd()
os.chdir("..\Files")

img = cv.imread(os.getcwd() + "/ape2.png")
os.chdir(oldwd)

# %%
# Converting to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)

# Blur
blur = cv.GaussianBlur(gray, (7, 7), cv.BORDER_DEFAULT)
cv.imshow("Blur_gray", blur)

# Edge Cascade: Detection of edges present in the images
canny_b = cv.Canny(blur, 125, 175)
canny = cv.Canny(img, 125, 175)

cv.imshow("canny_blur", canny_b)
cv.imshow("canny", canny)

# Dilating the image
dilated = cv.dilate(canny_b, (7, 7), iterations=1)

cv.imshow("dilated", dilated)
cv.imshow("canny", canny_b)

# Eroding the image

eroded = cv.erode(dilated, (3, 3), iterations=1)
cv.imshow("eroded", eroded)

# Resize
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_LINEAR)
cv.imshow("re", resized)

# Cropping : can be done by array slicing image array
cropped = img[100:400, 200:400]
cv.imshow("crop", cropped)
cv.waitKey(0)
