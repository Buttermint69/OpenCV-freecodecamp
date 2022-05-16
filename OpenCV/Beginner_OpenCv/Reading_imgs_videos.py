import cv2 as cv
import os

"""
# Reading images
"""
oldwd = os.getcwd()
os.chdir("..\Files")

img = cv.imread(os.getcwd() + "/ape2.png")

cv.imshow("Ape", img)
cv.waitKey(0)

# %%
"""
# Reading videos
"""

# %%
capture = cv.VideoCapture(os.getcwd() + "/image_1.gif")
os.chdir(oldwd)

try:
    while True:
        isTrue, frame = capture.read()
        cv.imshow("vid", frame)

        if cv.waitKey(20) & 0xFF == ord("d"):
            break
except Exception:
    capture.release()
    cv.destroyAllWindows()
