import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from miscellaneous import imshow


# 1.2
# hitogram equalisation
def my_equalizeHist(img, name):
    num_bins = 256
    plt.clf()
    # Calculate the histogram for src
    imag_hist, bins = np.histogram(img.flatten(), num_bins, density=True)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.savefig(name + '_hist_org.png')
    plt.clf()
    # Return the cumulative sum of the elements
    cdf = imag_hist.cumsum()

    # Normalize the histogram so that the sum of histogram bins is 255

    # numerator & denominator
    nj = (cdf - cdf.min()) * 255
    N = cdf.max() - cdf.min()
    # re-normalize the cumsum
    cdf = nj / N
    # cast it back to uint8 since we can't use floating point values in images
    cdf = cdf.astype('uint8')
    plt.plot(cdf)
    plt.savefig(name + '_cdf_org.png')
    plt.clf()

    eq_img = np.interp(img.flatten(), bins[:-1], cdf)
    eq_img = eq_img.reshape(img.shape)
    imag_hist_eq, _ = np.histogram(eq_img.flatten(), num_bins, density=True)
    plt.hist(imag_hist_eq.ravel(), 256, [0, 256])
    plt.savefig(name + '_hist_eq.png')
    plt.clf()

    cdf = imag_hist_eq.cumsum()
    # numerator & denomenator
    nj = (cdf - cdf.min()) * 255
    N = cdf.max() - cdf.min()
    # re-normalize the cumsum
    cdf = nj / N
    # cast it back to uint8 since we can't use floating point values in images
    cdf_eq = cdf.astype('uint8')
    plt.plot(cdf_eq)
    plt.savefig(name + '_cdf_eq.png')
    plt.clf()
    return eq_img

def hist_eq(img, gray, name):

    # enhance by histogram equalization and img is 2D
    if gray:
        #eq_img = cv.equalizeHist(img)
        eq_img2 = my_equalizeHist(img, name)

    else:
        b, g, r = cv.split(img)
        red = my_equalizeHist(r, name)
        blue = my_equalizeHist(b, name)
        green = my_equalizeHist(g, name)
        eq_img2 = cv.merge((blue, green, red))

    return eq_img2



# 1.1
path1 = 'resources/brain.jpg'
path2 = 'resources/brain_darker.jpg'
path3 = 'resources/nasa.jpg'

brain_light = cv.imread(path1)
brain_dark = cv.imread(path2)
nasa_colored = cv.imread(path3)

##convert to gray 1,2 and convert output nasa to RGB from BGR
brain_light = cv.cvtColor(brain_light, cv.COLOR_BGR2GRAY)
brain_dark = cv.cvtColor(brain_dark, cv.COLOR_BGR2GRAY)
#nasa_colored = cv.cvtColor(nasa_colored, cv.COLOR_BGR2RGB)

# 1.3

eq_br_light = hist_eq(brain_light, 1, 'results/brain_light')
path_out = 'brain_light_enhanced'
cv.normalize(eq_br_light, eq_br_light, 0, 255, cv.NORM_MINMAX)
imshow(eq_br_light, path_out)
eq_br_dark = hist_eq(brain_dark, 1, 'results/brain_dark')
path_out = 'brain_dark_enhanced'
cv.normalize(eq_br_dark, eq_br_dark, 0, 255, cv.NORM_MINMAX)
imshow(eq_br_dark, path_out)
nasa_separate = hist_eq(nasa_colored, 0, 'results/nasa_brg')  # each of its three channels separately
path_out = 'nasa_separate'
cv.normalize(nasa_separate, nasa_separate, 0, 1, cv.NORM_MINMAX)
imshow(nasa_separate, path_out)
nasa_colored_rgb = cv.cvtColor(nasa_colored, cv.COLOR_BGR2RGB)
nasa_separate_rgb = hist_eq(nasa_colored_rgb, 0, 'results/nasa_rgb')  # each of its three channels separately
path_out = 'nasa_separate_rgb'
cv.normalize(nasa_separate_rgb, nasa_separate_rgb, 0, 1, cv.NORM_MINMAX)
imshow(nasa_separate_rgb, path_out)


# 1.4

nasa_hsv = cv.cvtColor(nasa_colored, cv.COLOR_BGR2HSV)
nasa_hsv_v = nasa_hsv[:, :, 2]
eq_nasa_hsv_v = hist_eq(nasa_hsv_v, 1, 'results/nasa_hsv')
nasa_hsv[:, :, 2] = eq_nasa_hsv_v
nasa_enhanced = cv.cvtColor(nasa_hsv, cv.COLOR_HSV2BGR)
path_out = 'nasa_enhanced'
imshow(nasa_enhanced, path_out)

