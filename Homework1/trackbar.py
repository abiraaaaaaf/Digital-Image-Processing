import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse


alpha_slider_max = 100
title_window = 'Linear Blend'
list = []

def fourietransform(img, alpha):

    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # LPF
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    c_w = int(alpha / 2 * ccol)
    r_w = int(alpha / 2 * crow)

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - r_w: crow * r_w, ccol - c_w: ccol + c_w] = 1
    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back




def on_trackbar(val):
    alpha = val / alpha_slider_max
    for img in list:
        img_back = fourietransform(img, alpha)
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img_back, cmap='gray')
        plt.title('Output Image'), plt.xticks([]), plt.yticks([])
        plt.show()



# read images and store in a list

for i in range(3):
    filename = 'q3_images/Pic%d.jpg' %(i+1)
    img = cv.imread(filename, 0)
    list.append(img)
    cv.namedWindow(title_window)
    trackbar_name = 'Alpha x %d' % alpha_slider_max
    cv.createTrackbar(trackbar_name, title_window, 0, alpha_slider_max, on_trackbar)
    # Show some stuff
    on_trackbar(0)
    # Wait until user press some key
    cv.waitKey()

    cv.destroyAllWindows()







