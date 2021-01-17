import numpy as np

class Interpolation:

    @staticmethod
    def avg_interpolate(img_array):
        row, col, _ = img_array.shape
        # Filling With Zeros The Out Array
        new_img_array = np.zeros((row, col*2, 3))

        # first assigning image pixels to the even cells in new array
        for i in range(row):
            for j in range(col):
               new_img_array[i][2 * j][:] = img_array[i][j][:]

        # then filling the odd pixels with average two nearest row pixels (it can be the other two column pixels or four or 8 pixels ^_^)
        for i in range(row):
            for j in range(col):
                new_img_array[i][2 * j + 1][:] = (new_img_array[i][2 * j][:] + new_img_array[i][2 * j - 1])/2

        return new_img_array
        pass