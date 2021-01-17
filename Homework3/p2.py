import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from skimage import restoration
from miscellaneous import imshow, sum_of_absolutes



path = 'resources/azadi.jpg'
img = cv.imread(path)
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
plt.imshow(img, cmap='gray')

psf = np.array(
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
)
psf = psf / np.sum(psf)


# ??? convolved = ... (use 'signal' module to convolve img with psf. The output's shape must have the same shape as img)
convolved = signal.convolve(img, psf, 'same')

convolved /= 255
print(convolved.min(), convolved.max())
imshow(convolved)


# ??? deconvolved = ... (use restoration module to estimate img using richardson_lucy method)
deconvolved = restoration.richardson_lucy(convolved, psf, iterations=30)
print(deconvolved.min(), deconvolved.max())
print(deconvolved)
cv.normalize(deconvolved, deconvolved, 0, 255, cv.NORM_MINMAX)
print(deconvolved)
print(deconvolved.min(), deconvolved.max())
imshow(deconvolved, 'richardson_lucy_estimation')
print(sum_of_absolutes(img, deconvolved))


# ??? deconvolved_w = ... (use restoration module to estimate img using unsupervised_wiener method)
deconvolved_w, _ = restoration.unsupervised_wiener(convolved, psf)
cv.normalize(deconvolved_w, deconvolved_w, 0, 255, cv.NORM_MINMAX, )
print(deconvolved_w)
imshow(deconvolved_w, 'wiener_estimation')
print(sum_of_absolutes(img, deconvolved_w))



### PART 2 ###

# convolved_f = ??? FFT of 'convolved' variable using numpy
convolved_f = np.fft.fft2(convolved)


# psf_f = ??? FFT of 'psf' variable using numpy
psf_f = np.fft.fft2(psf, s=(1080, 1920)) # s


# deconvolved_i_f = ??? multiply convolved_f and 1/(1 + psf_f) in Fourier domain to convolve in space domain
# 1 in the denominator is to avoid division by zero.
# Change 1 to any positive number if you think that would make your results better.
deconvolved_i_f = convolved_f/(1 + psf_f)

# deconvolved_i = ???  apply inverse FFT to deconvolved_i_f
# ??? trow out the imaginary part of deconvolved_i matrix


deconvolved_i = np.fft.ifft2(deconvolved_i_f)

deconvolved_i = np.abs(deconvolved_i)
print(deconvolved.min(), deconvolved.max())


# Here you should adjust the range of values in deconvolved_i, and then convert it to 8-bit unsigned integer.
# Also, print min and max of deconvolved_i before converting to integer.
# ???
cv.normalize(deconvolved_i, deconvolved_i, 0, 255, cv.NORM_MINMAX)
imshow(deconvolved_i, 'inverse_filtered')
# print(sum_of_absolutes(deconvolved_i, img)
