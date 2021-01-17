import cv2
import numpy as np
from ipl_utils import interpolate

list = []
# read the 3 images and append to list
for i in range(3):
    filename = 'q2_images/Pic%d.jpg'%(i+1)
    img = cv2.imread(filename)
    list.append(img)
i = 0

# interpolate all the images in list and store widened images :)
for image in list:
    interpolated_img = interpolate.Interpolation.avg_interpolate(image)
    filename_interpolated = 'q2_images/Pic_interpolated%d.jpg' % (i + 1)
    cv2.imwrite(filename_interpolated, interpolated_img)
    print('Main Image', i, image.shape)
    print('Interpolated Image', i, interpolated_img.shape)
    i = i + 1
