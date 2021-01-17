import cv2 as cv
import numpy as np
import math


# 2.1
print('-'*100)
print('part 2.1 ')

path = 'resources/images/sudoku.png'
original = cv.imread(path)
img = cv.cvtColor(original, cv.COLOR_BGR2GRAY)



# equalizeHist
equ = cv.equalizeHist(img)
cv.imwrite('results/Q2/equ.png', equ)

# CLAHE

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_he = clahe.apply(img)
cv.imwrite('results/Q2/img_he.png', img_he)

# 2.2
print('-'*100)
print('part 2.2 ')

def funcCan(thresh1=0):
    thresh1 = cv.getTrackbarPos('thresh1', 'canny')
    thresh2 = cv.getTrackbarPos('thresh2', 'canny')
    edge = cv.Canny(img, thresh1, thresh2)
    cv.imshow('canny', edge)

cv.namedWindow('canny')
thresh1 = 100
thresh2 = 1
cv.createTrackbar('thresh1', 'canny', thresh1, 255, funcCan)
cv.createTrackbar('thresh2', 'canny', thresh2, 255, funcCan)
funcCan(0)

cv.imshow('Frame', original)
cv.waitKey(0)

best_low_threshold = 34
best_high_threshold = 109
edges = cv.Canny(img, best_low_threshold, best_high_threshold, apertureSize = 3)

cv.imwrite('results/Q2/canny.png', edges)

# 2.3
print('-'*100)
print('part 2.3')

cdst = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)
lines = cv.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

cv.imshow("Source", img)
cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
cv.imwrite('results/Q2/HoughLines.png', cdst)
cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
cv.imwrite('results/Q2/HoughLinesP.png', cdstP)
cv.waitKey()

cv.destroyAllWindows()