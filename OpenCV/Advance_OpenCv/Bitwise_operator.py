import cv2 as cv
import numpy as np
"""
# Bitwise Operators
"""

# %%
blank = np.zeros((400, 400), dtype="uint8")
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)
cv.imshow("rectangle", rectangle)
cv.imshow("circle", circle)

# Bitwise AND
# A ---> 0, 0, 1, 1
# AND
# B ---> 0, 1, 0, 1
# -----> 0, 0, 0, 1

bit_AND = cv.bitwise_and(rectangle, circle)
cv.imshow("AND", bit_AND)

# Bitwise OR
# A ---> 0, 0, 1, 1
# OR
# B ---> 0, 1, 0, 1
# -----> 0, 1, 1, 1

bit_OR = cv.bitwise_or(rectangle, circle)
cv.imshow("OR", bit_OR)

# Bitwise XOR --> non-intersecting region
bit_XOR = cv.bitwise_xor(rectangle, circle)
cv.imshow("XOR", bit_XOR)

# Bitwise NOT 0--->1
bit_NOT = cv.bitwise_not(rectangle)
cv.imshow("NOT", bit_NOT)

cv.waitKey(0)