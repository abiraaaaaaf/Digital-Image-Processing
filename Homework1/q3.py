import cv2
import numpy as np
from matplotlib import pyplot as plt
from fourier_Transform import fourier

# read images and store in a list
list = []

for i in range(3):
    filename = 'q3_images/Pic%d.jpg' %(i+1)
    img = cv2.imread(filename, 0)
    list.append(img)

# Fourier Transform
i = 1
for img in list:
    print(img.shape)
    print('Image Number', i)
    # Fourier Transform Function
    fourier(img, i)
    i = i + 1