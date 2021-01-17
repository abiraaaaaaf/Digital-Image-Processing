import numpy as np
import cv2
import skimage
import math
from PIL import Image
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet
from scipy.ndimage import median_filter
from cv2 import fastNlMeansDenoisingColored
from skimage.measure import compare_psnr
from matplotlib import pyplot as plt


def psnr(im1, im2):
    psnr_value = compare_psnr(im1, im2, data_range=(np.max(im1)-np.min(im1)))
    print("PSNR IS : ", psnr_value)

# 3.1
print('-'*100)
print('part 3.1 ')

path = 'resources/images/fantasy.jpg'
img = Image.open(path)

size = list(img.size)
size[0] /= 3
size[1] /= 3
size[0] = int(size[0])
size[1] = int(size[1])
out = img.resize(size, Image.NEAREST)
out.save("results/Q3/smaller_image_fantasy.png")

print(out.size)

#3.2
print('-'*100)
print('part 3.2')

# convert PIL Image to ndarray
im_arr = np.asarray(out)
noise_img = random_noise(im_arr, mode='gaussian',var=0.0256)
noise_img = (255*noise_img).astype(np.uint8)

img_ga = Image.fromarray(noise_img)
img_ga.save('results/Q3/img_ga.jpg')

noise_img = random_noise(im_arr, mode='speckle', var=0.6)
noise_img = (255*noise_img).astype(np.uint8)

img_mult = Image.fromarray(noise_img)
img_mult.save('results/Q3/img_mult.jpg')

noise_img = random_noise(im_arr, mode='s&p', salt_vs_pepper=0.55, amount=0.06)  # salt 0.55 pepper 0.45
noise_img = (255*noise_img).astype(np.uint8)
img_sp = Image.fromarray(noise_img)

img_sp.save('results/Q3/img_sp.jpg')

# 3.3
print('-'*100)
print('part 3.3 ')

img_ga = np.asanyarray(img_ga)

img_ga_tvc = denoise_tv_chambolle(img_ga, weight=0.1, multichannel=True)
cv2.normalize(img_ga_tvc, img_ga_tvc, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_ga_tvc)
img_ga_tvc = Image.fromarray(np.uint8(img_ga_tvc))
img_ga_tvc.save("results/Q3/img_ga_tvc.jpg")

img_ga_bil = denoise_bilateral(img_ga, win_size=10, multichannel=True)
cv2.normalize(img_ga_bil, img_ga_bil, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_ga_bil)
img_ga_bil = Image.fromarray(np.uint8(img_ga_bil))
img_ga_bil.save("results/Q3/img_ga_bil.jpg")

img_ga_wav = denoise_wavelet(img_ga, multichannel=True, sigma=0.1)
cv2.normalize(img_ga_wav, img_ga_wav, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_ga_wav)
img_ga_wav = Image.fromarray(np.uint8(img_ga_wav))
img_ga_wav.save("results/Q3/img_ga_wav.jpg")

img_ga_med = median_filter(img_ga, size=2)
cv2.normalize(img_ga_med, img_ga_med, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_ga_med)
img_ga_med = Image.fromarray(np.uint8(img_ga_med))
img_ga_med.save("results/Q3/img_ga_med.jpg")

img_ga_cvd = fastNlMeansDenoisingColored(img_ga, hColor=10, templateWindowSize=7, h=14)
cv2.normalize(img_ga_cvd, img_ga_cvd, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_ga_cvd)
img_ga_cvd = Image.fromarray(np.uint8(img_ga_cvd))
img_ga_cvd.save("results/Q3/img_ga_cvd.jpg")

img_mult = np.asanyarray(img_mult)

img_mult_tvc = denoise_tv_chambolle(img_mult, weight=0.15, multichannel=True)
cv2.normalize(img_mult_tvc, img_mult_tvc, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_mult_tvc)
img_mult_tvc = Image.fromarray(np.uint8(img_mult_tvc))
img_mult_tvc.save("results/Q3/img_mult_tvc.jpg")

img_mult_bil = denoise_bilateral(img_mult, multichannel=True)
cv2.normalize(img_mult_bil, img_mult_bil, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_mult_bil)
img_mult_bil = Image.fromarray(np.uint8(img_mult_bil))
img_mult_bil.save("results/Q3/img_mult_bil.jpg")

img_mult_wav = denoise_wavelet(img_mult, multichannel=True, sigma=0.22, method='BayesShrink', wavelet_levels=4)
cv2.normalize(img_mult_wav, img_mult_wav, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_mult_wav)
img_mult_wav = Image.fromarray(np.uint8(img_mult_wav))
img_mult_wav.save("results/Q3/img_mult_wav.jpg")

img_mult_med = median_filter(img_mult, size=3)
cv2.normalize(img_mult_med, img_mult_med, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_mult_med)
img_mult_med = Image.fromarray(np.uint8(img_mult_med))
img_mult_med.save("results/Q3/img_mult_med.jpg")

img_mult_cvd = fastNlMeansDenoisingColored(img_mult, hColor=10, h=27)
cv2.normalize(img_mult_cvd, img_mult_cvd, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_mult_cvd)
img_mult_cvd = Image.fromarray(np.uint8(img_mult_cvd))
img_mult_cvd.save("results/Q3/img_mult_cvd.jpg")

img_sp = np.asanyarray(img_sp)

img_sp_tvc = denoise_tv_chambolle(img_sp, weight=0.2, multichannel=True)
cv2.normalize(img_sp_tvc, img_sp_tvc, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_sp_tvc)
img_sp_tvc = Image.fromarray(np.uint8(img_sp_tvc))
img_sp_tvc.save("results/Q3/img_sp_tvc.jpg")

img_sp_bil = denoise_bilateral(img_sp, multichannel=True)
cv2.normalize(img_sp_bil, img_sp_bil, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_sp_bil)
img_sp_bil = Image.fromarray(np.uint8(img_sp_bil))
img_sp_bil.save("results/Q3/img_sp_bil.jpg")

img_sp_wav = denoise_wavelet(img_sp, multichannel=True, sigma=0.13)
cv2.normalize(img_sp_wav, img_sp_wav, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_sp_wav)
img_sp_wav = Image.fromarray(np.uint8(img_sp_wav))
img_sp_wav.save("results/Q3/img_sp_wav.jpg")

img_sp_med = median_filter(img_sp, 3)
cv2.normalize(img_sp_med, img_sp_med, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_sp_med)
img_sp_med = Image.fromarray(np.uint8(img_sp_med))
img_sp_med.save("results/Q3/img_sp_med.jpg")

img_sp_cvd = fastNlMeansDenoisingColored(img_sp, hColor=10, h=27)
cv2.normalize(img_sp_cvd, img_sp_cvd, 0, 255, cv2.NORM_MINMAX)
psnr(im_arr, img_sp_cvd)
img_sp_cvd = Image.fromarray(np.uint8(img_sp_cvd))
img_sp_cvd.save("results/Q3/img_sp_cvd.jpg")

cv2.waitKey(0)
