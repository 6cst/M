pip install numpy

import numpy as np

import sys
list_1000000 = list(range(1000000))
array_1000000 = np.arange(1000000)
print(f"Size of list: {sys.getsizeof(list_1000000)} bytes")
print(f"Size of array: {array_1000000.nbytes} bytes")
import time
start_time = time.time()
sum_list = sum(list_1000000)
end_time = time.time()
list_sum_time = (end_time - start_time) * 1000
start_time = time.time()
sum_array = np.sum(array_1000000)
end_time = time.time()
array_sum_time = (end_time - start_time) * 1000
print(f"Time to sum list: {list_sum_time:.2f} milliseconds")
print(f"Time to sum array: {array_sum_time:.2f} milliseconds")

# Create an array from 0 to 9
arr = np.arange(10)
print(arr)  # Output: [0 1 2 3 4 5 6 7 8 9]

# Create a 1D array
arr_1d = np.array([1, 2, 3, 4, 5])
print(arr_1d)       # Output: [1 2 3 4 5]
print(arr_1d[0])    # Output: 1

# Create a 2D array
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(arr_2d)

# Create a 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Print the shape
print("Shape of the array:", arr_2d.shape)  # Output: (2, 3)

# Create an array with a specific data type
arr_float = np.array([1.1, 2.2, 3.3], dtype=np.float32)
# Print the data type
print("Data type of the array:", arr_float.dtype)  # Output: float32

# Create a 3D array
arr_3d = np.zeros((2, 3, 4))
print(arr_3d)

# Print the size
print("Size of the array:", arr_3d.size)  # Output: 24 (2 * 3 * 4)

# Create a 4D array
arr_4d = np.ones((2, 3, 4, 5))

# Print the number of dimensions
print("Number of dimensions:", arr_4d.ndim)  # Output: 4

# Create an array of integers
arr_int = np.array([1, 2, 3])
print(arr_int.dtype)
# Print the item size
print("Size of each element (in bytes):", arr_int.itemsize)  # Output: 8 (for 64-bit integer)

# Create an array with a specific data type
arr_int32 = np.array([1, 2, 3], dtype=np.int32)
arr_float64 = np.array([1.1, 2.2, 3.3], dtype=np.float64)

print("Data type of arr_int32:", arr_int32.dtype)   # Output: int32
print("Data type of arr_float64:", arr_float64.dtype)   # Output: float64

# Create an array with different data types
arr_int32 = np.array([1234567890, 1234567890], dtype=np.int32)
arr_int64 = np.array([1234567890, 1234567890], dtype=np.int64)

print("Data type of arr_int32:", arr_int32.dtype)   # Output: int32
print("Data type of arr_int64:", arr_int64.dtype)   # Output: int64

print(arr_int32)

# In this example, the `int32` data type has limited precision compared to `int64`, which can represent larger integers without loss of precision.


