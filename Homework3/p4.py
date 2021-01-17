import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from miscellaneous import imshow


path1 = 'resources/wiki.jpg'
# img = ... (read the image using OpenCV using variable 'path')
img = cv.imread(path1)
start_row, start_col = int(0), int(0)
y_end = int(720)
x_end = int(1920/2)
img_l_diff = img[int(0): y_end, int(0):x_end]
print(img.shape)  # (720, 1920, 3)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img_l = img[int(0): y_end, int(0):x_end]
img_r = img[int(0): y_end, x_end:int(1920)]

print(img_l.shape)
print(img_r.shape)
imshow(img_l, 'img_l')
imshow(img_r, 'img_r')

diff = cv.subtract(img_r, img_l)


for i in range(720):
    for j in range(960):
        if(diff[i,j] != 0):
            img_l_diff[i, j, 0] = 0
            img_l_diff[i, j, 1] = 0
            img_l_diff[i, j, 2] = 255


imshow(diff, "differences")
imshow(img_l_diff, "wiki_differences")


