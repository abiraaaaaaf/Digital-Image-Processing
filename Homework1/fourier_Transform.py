import cv2
import numpy as np
from matplotlib import pyplot as plt


def fourier(image, i):
    #fourier DFT
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum, phase_spectrum = 20 * np.log(cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # plt.subplot(121), plt.imshow(image, cmap='gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # Low Pass Filter
    rows, cols = image.shape
    crow, ccol = int(rows / 2), int(cols / 2)   # Find middle and convert to int
    a = [1 / 10, 1 / 5, 1 / 2]
    for c_w, r_w in zip(a, a):
        cnt = int(1 / c_w)
        print('LPF preserve: ', c_w)
        # e.g. if c_w is 1/10 then 1/20 of the dft image from left and 1/20 from right is selected for LPF
        c = int(c_w / 2 * ccol)
        r = int(r_w / 2 * crow)

        # create a mask first, center square is 1, remaining all zeros
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - r: crow + r, ccol - c: ccol + c] = 1   # outer space of this mask is zero so high frequencies are not considered :)

        # apply mask and inverse DFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        # Plot img_back output inverse DFT
        plt.subplot(121), plt.imshow(image, cmap='gray')
        plt.title('Input Image Fourier'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img_back, cmap='gray')
        plt.title('Output Image Fourier'), plt.xticks([]), plt.yticks([])
        plt.show()
        filename = 'q3_images/Pic_IDF %d.jpg' % (i + 1)
        plt.imsave(filename, img_back)
