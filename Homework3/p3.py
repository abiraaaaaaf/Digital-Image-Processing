import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import miscellaneous

# 3.1
def quantize(img, num_levels):

    span = 256 / num_levels  # 51
    dim1, dim2 = img.shape
    img_int = img.astype(np.int64)  # convert unit8 to int
    out_img = np.zeros((dim1, dim2))  # create output matrix
    out_img_int = out_img.astype(np.int64)
    for i in range(0, dim1):
        for j in range(0, dim2):
                out_img_int[i][j] = np.floor((np.floor(img_int[i][j] / span)) * span + (span / 2))

    result = out_img_int.astype(np.uint8)
    assert np.unique(result).size == num_levels, 'quantizer is broken'
    return (result)

# load image 1
mobile1 = cv.imread('resources/mobile1.jpg', 0)
mobile1_q = quantize(mobile1, 5)
cv.imwrite('results/mobile1_q.jpg', mobile1_q)
# print(miscellaneous.sum_of_absolutes(mobile1,mobile1q)) #print sum_of_absolute error

# load image2

mobile2 = cv.imread('resources/mobile2.jpg',0)
mobile2_q = quantize(mobile2, 5)
cv.imwrite('results/mobile2_q.jpg', mobile2_q)
# print(miscellaneous.sum_of_absolutes(mobile2, mobile2q)) #print sum_of_absolute error


# 3.2

hnd = cv.imread('resources/hnd.jpg')
hnd = cv.cvtColor(hnd, cv.COLOR_BGR2GRAY)
hnd = hnd.astype(np.int64)
int_out = np.zeros((8, 1096, 818))
out = np.zeros((1096, 818))
diff = np.zeros((8))

for k in range(8):
    pos = k
    for i in range(1096):
        for j in range(818):
            bin = '{0:08b}'.format(hnd[i, j])
            temp = bin
            temp = temp[:pos] + '0' + temp[(pos + 1):]
            # print(bin)
            # print(temp)
            int_out[k, i, j] = int(temp, 2)
            # print(int_out[k, i, j])
    out[:, :] = int_out[k, :, :]
    out = out.astype(np.int64)

    # print(out.dtype)
    # print(out.shape)
    # print(hnd.dtype)
    # print(hnd.shape)

    diff[k] = miscellaneous.sum_of_absolutes(out,  hnd)

print(diff)
minimum = np.min(diff)
print('min is:', minimum)

for i in range(8):
    if minimum == diff[i]:
        k = i
print(k)

for i in range(1096):
    for j in range(818):
        bin = '{0:08b}'.format(hnd[i, j])
        temp = bin
        temp = temp[:pos] + '0' + temp[(pos + 1):]
        # print(bin)
        # print(temp)
        # print('.............................')
        int_out[k, i, j] = int(temp, 2)

out[:, :] = int_out[k, :, :]
out = out.astype(np.uint8)
hnd_q = quantize(out, 5)

cv.imwrite('results/hnd_q.jpg', hnd_q)
