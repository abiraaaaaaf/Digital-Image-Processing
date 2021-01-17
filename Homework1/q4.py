import cv2 as cv
import numpy as np

# read images
give_red = cv.imread('q4_images/give_red.jpg')
img_main = cv.imread('q4_images/base.jpg')
img_op = cv.imread('q4_images/base.jpg')

#4.main

# Red channel give_red to red channel base
img_main[:, :, 2] = give_red[:, :, 2]
# remove blue and green
give_red[:, :, 0] = 0
give_red[:, :, 1] = 0
# print(b)
cv.imwrite('q4_images/red_RGB.jpg', give_red)   # Red Part Of the Image give_red
cv.imwrite('q4_images/Base_red.jpg', img_main)  # Red channel give_red to red channel base in part main


#4.optional
print(give_red.shape)
row, col, _ = give_red.shape
for i in range(row):
    for j in range(col):
        # print(b[i, j, 2])
        # Red channel give_red to red channel base when the value is higher than 95 Threshold Experimenti Entekhab kardam!!! :|
        if give_red[i, j, 2] > 95:
            img_op[i, j, 2] = give_red[i, j, 2]

cv.imwrite('q4_images/Base_red_optional.jpg', img_op) # Red channel give_red to red channel base in part optional


