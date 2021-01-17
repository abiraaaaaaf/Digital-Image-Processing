import numpy as np


class MatrixOperations:
    """
    A collection of methods to perform matrix operations.
    """

    @staticmethod
    def kron(a, b):

        """Compute kronecker product of a and b, and return it."""

        row_a, col_a = np.shape(a)
        row_b, col_b = np.shape(b)

        kronecker_a_b = np.zeros((row_a*row_b, col_a*col_b))   # matrix initialization with zero
        for i in range(row_a):
            for j in range(row_b):
                for k in range(col_a):
                    for l in range(col_b):
                        kronecker_a_b[i*row_b+j][k*col_b+l]=a[i][k]*b[j][l]
        return kronecker_a_b

        pass
