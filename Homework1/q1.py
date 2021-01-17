import numpy as np
from ipl_utils import ipl_math

# Kronecker product between two matrices

# pair 1  two integer matrices
rand_int_5 = np.random.randint(0, 255, (5, 5))
rand_int_2 = np.random.randint(0, 255, (2, 2))

p1_ipl_func = ipl_math.MatrixOperations.kron(rand_int_5, rand_int_2)
p1_numpy_func = np.kron(rand_int_5, rand_int_2)  # use np.kron to assign to this variable

assert np.array_equal(p1_ipl_func, p1_numpy_func)
print("No Problem In Assert Integer Kron :D")
# how to print some information about a numpy variable
# print(rand_int_5.shape)
# print(type(rand_int_5))
# print(rand_int_5.dtype)

# pair 2  float
rand_float_5 = np.random.uniform(0, 255, (5, 5))  # 5 by 5 matrix with float data-types
rand_float_2 = np.random.uniform(0, 255, (2, 2))  # 2 by 2 matrix with float data-types

p2_ipl_func = ipl_math.MatrixOperations.kron(rand_float_5, rand_float_2)
p2_numpy_func = np.kron(rand_float_5, rand_float_2)  # use np.kron to assign to this variable

# print(p2_ipl_func.shape)
# print(p2_numpy_func.shape)

assert np.array_equal(p2_ipl_func, p2_numpy_func)
print("No Problem In Assert Float Kron :D")


# Just For testing astype convert to int64
p2_numpy_func_int = p2_numpy_func.astype(np.int64)
p2_ipl_func_int = p2_ipl_func.astype(np.int64)


assert np.array_equal(p2_ipl_func_int, p2_numpy_func_int)
print("No Problem In Assert Float (With Using Astype!) Kron :D")


# pair 3  normal distribution
mu = 0
sigma = 0.1
rand_normal_5 = np.random.normal(mu, sigma, (5, 5))  # 5 by 5 matrix, values sampled from a Normal distribution
rand_normal_2 = np.random.normal(mu, sigma, (2, 2))  # 2 by 2 matrix, values sampled from a Normal distribution

p3_ipl_func = ipl_math.MatrixOperations.kron(rand_normal_5, rand_normal_2)
p3_numpy_func = np.kron(rand_normal_5, rand_normal_2)  # use np.kron to assign to this variable

assert np.array_equal(p3_ipl_func, p3_numpy_func)
print("No Problem In Assert Normal Distribution Kron :D")

# pair 4  all elements equal to 1
ones_5 = np.ones_like(rand_int_5)
twos_2 = 2 * ones_5

p4_ipl_func = ipl_math.MatrixOperations.kron(ones_5, twos_2)
p4_numpy_func = np.kron(ones_5, twos_2)

assert np.array_equal(p4_ipl_func, p4_numpy_func)
print("No Problem In Assert All Elements Equal to Ones Kron :D")
