import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def imshow(img, name=None):
    plt.imshow(img, cmap='gray')
    if name is not None:
        path = './results/'+name+'.jpg'
        cv.imwrite(path, img)


def sum_of_absolutes(m1, m2):
    """
    Return average difference of values of m1 and m2 per pixel.
    """

    diff = m1 - m2  # elementwise for scipy arrays
    dif = np.array(diff).ravel()
    m_norm = np.sum(np.abs(dif))
    return m_norm
    pass
