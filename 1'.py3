#NumPy


NumPy is a fundamental package for numerical computing in Python. It provides support for multidimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently. Key features of NumPy include:

1. **N-dimensional Arrays**: NumPy's main object is the homogeneous multidimensional array called ndarray. These arrays can have any number of dimensions and contain elements of the same data type.

2. **Efficient Operations**: NumPy provides a wide range of mathematical functions that operate element-wise on arrays, enabling efficient numerical computations.

3. **Broadcasting**: NumPy's broadcasting capability allows operations between arrays of different shapes and sizes without explicitly iterating over them, which can greatly simplify code and improve performance.

4. **Indexing and Slicing**: NumPy offers powerful indexing and slicing capabilities for accessing and modifying elements within arrays, including advanced techniques like boolean indexing and fancy indexing.

5. **Integration with Other Libraries**: NumPy is the foundation for many other scientific computing libraries in Python, such as SciPy, Matplotlib, and scikit-learn. It seamlessly integrates with these libraries, providing a cohesive ecosystem for scientific computing.

**Importing NumPy:**

import numpy as np



---



# NumPy vs Python


NumPy arrays and Python lists are both used to store collections of data, but they have some fundamental differences in terms of functionality, performance, and memory usage. Here's a comparison between NumPy arrays and Python lists:

1. **Functionality:**
   - NumPy arrays: NumPy arrays are homogeneous, meaning all elements in the array are of the same data type. They support a wide range of mathematical operations and functions, such as element-wise operations, linear algebra, Fourier transforms, random number generation, and more. NumPy also provides advanced indexing and slicing capabilities.
   - Python lists: Python lists can contain elements of different data types, making them heterogeneous. They offer basic functionality for iterating over elements, appending, inserting, and removing elements, but they lack specialized mathematical operations.

2. **Performance:**
   - NumPy arrays: NumPy arrays are highly optimized for numerical operations and are implemented in C, which makes them significantly faster than Python lists for numerical computations, especially when working with large datasets.
   - Python lists: Python lists are implemented in Python itself, which can lead to slower performance compared to NumPy arrays, particularly for numerical computations involving large datasets.

3. **Memory Usage:**
   - NumPy arrays: NumPy arrays typically use less memory compared to Python lists, especially for large datasets, because they store data in a contiguous block of memory and can take advantage of data type optimizations.
   - Python lists: Python lists can potentially use more memory compared to NumPy arrays due to the overhead of storing additional information about each element (e.g., type information, reference count).

4. **Ease of Use:**
   - NumPy arrays: NumPy arrays offer a wide range of functions and methods specifically designed for numerical computing tasks, making them convenient to use for scientific and mathematical applications.
   - Python lists: Python lists are more general-purpose and flexible, making them easier to work with for tasks that do not involve numerical computations or require heterogeneous data types.

import sys

# Create a list of 1000000 integers
list_1000000 = list(range(1000000))

# Create a NumPy array of 1000000 integers
array_1000000 = np.arange(1000000)

# Print the size of the list and the array in bytes
print(f"Size of list: {sys.getsizeof(list_1000000)} bytes")
print(f"Size of array: {array_1000000.nbytes} bytes")

# Time how long it takes to sum the list and the array
import time

# Sum the list
start_time = time.time()
sum_list = sum(list_1000000)
end_time = time.time()
list_sum_time = (end_time - start_time) * 1000

# Sum the array
start_time = time.time()
sum_array = np.sum(array_1000000)
end_time = time.time()
array_sum_time = (end_time - start_time) * 1000

# Print the time it took to sum the list and the array
print(f"Time to sum list: {list_sum_time:.2f} milliseconds")
print(f"Time to sum array: {array_sum_time:.2f} milliseconds")




---



# NumPy Arrays


**np.arange():**

np.arange() is a function in NumPy used to create an array with regularly spaced values within a specified range. Its syntax is:

 # numpy.arange([start, ]stop, [step, ]dtype=None)


- `start`: Optional. The start of the interval (inclusive). Default is 0.
- `stop`: The end of the interval (exclusive).
- `step`: Optional. The step size between values. Default is 1.
- `dtype`: Optional. The data type of the array. If not specified, the data type is inferred from the other input arguments.

# Create an array from 0 to 9
arr = np.arange(10)
print(arr)  # Output: [0 1 2 3 4 5 6 7 8 9]

**NumPy Arrays (ndarrays):**

NumPy arrays, or ndarrays, are the primary data structure used in NumPy. They are homogeneous collections of elements with fixed dimensions and have many similarities to Python lists but with additional functionality optimized for numerical computing.

**1D Arrays:**

1D arrays are like traditional arrays or lists. They have a single row and can be indexed using a single index.

# Create a 1D array
arr_1d = np.array([1, 2, 3, 4, 5])
print(arr_1d)       # Output: [1 2 3 4 5]
print(arr_1d[0])    # Output: 1

**2D Arrays:**

2D arrays, also known as matrices, have rows and columns. They are indexed using two indices, one for the row and one for the column.

# Create a 2D array
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(arr_2d)



---



# Properties of nd Arrays

**1. Shape:**

The shape of an ndarray describes the size of each dimension of the array. It is represented as a tuple of integers indicating the number of elements along each dimension.

# Create a 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Print the shape
print("Shape of the array:", arr_2d.shape)  # Output: (2, 3)

**2. Data Type (dtype):**

The data type of an ndarray specifies the type of elements stored in the array. NumPy arrays can hold elements of different types such as integers, floats, booleans, etc.

# Create an array with a specific data type
arr_float = np.array([1.1, 2.2, 3.3], dtype=np.float32)
# Print the data type
print("Data type of the array:", arr_float.dtype)  # Output: float32

**3. Size:**

The size of an ndarray is the total number of elements in the array. It is equal to the product of the dimensions of the array.

# Create a 3D array
arr_3d = np.zeros((2, 3, 4))
print(arr_3d)

# Print the size
print("Size of the array:", arr_3d.size)  # Output: 24 (2 * 3 * 4)

**4. Number of Dimensions (ndim):**

The ndim attribute of an ndarray specifies the number of dimensions or axes of the array.

# Create a 4D array
arr_4d = np.ones((2, 3, 4, 5))

# Print the number of dimensions
print("Number of dimensions:", arr_4d.ndim)  # Output: 4

**5. Itemsize:**

The itemsize attribute of an ndarray specifies the size of each element in bytes.

# Create an array of integers
arr_int = np.array([1, 2, 3])
print(arr_int.dtype)
# Print the item size
print("Size of each element (in bytes):", arr_int.itemsize)  # Output: 8 (for 64-bit integer)



---





#NumPy Data Types and Precision

**NumPy Data Types (dtypes):**

NumPy provides a variety of data types to represent different kinds of numerical data. These data types are important for controlling memory usage and ensuring data integrity in numerical computations.

**Common NumPy Data Types:**

- **int**: Integer (default size depends on the platform).
- **float**: Floating point number (default size depends on the platform).
- **bool**: Boolean (True or False).
- **complex**: Complex number with real and imaginary parts.
- **uint**: Unsigned integer (no negative values).

**Specifying Data Types:**

You can specify the data type of an ndarray using the `dtype` parameter of NumPy functions or by providing the data type as an argument to the array creation functions.

# Create an array with a specific data type
arr_int32 = np.array([1, 2, 3], dtype=np.int32)
arr_float64 = np.array([1.1, 2.2, 3.3], dtype=np.float64)

print("Data type of arr_int32:", arr_int32.dtype)   # Output: int32
print("Data type of arr_float64:", arr_float64.dtype)   # Output: float64

**Precision:**

Precision refers to the level of detail and accuracy with which numerical values are represented. NumPy data types have different levels of precision, which determine the range of values they can represent and the amount of memory they occupy.

# Create an array with different data types
arr_int32 = np.array([1234567890, 1234567890], dtype=np.int32)
arr_int64 = np.array([1234567890, 1234567890], dtype=np.int64)

print("Data type of arr_int32:", arr_int32.dtype)   # Output: int32
print("Data type of arr_int64:", arr_int64.dtype)   # Output: int64

print(arr_int32)

# In this example, the `int32` data type has limited precision compared to `int64`, which can represent larger integers without loss of precision.

**Impact of Precision on Memory Usage:**

# Create arrays with different data types
arr_float32 = np.array([1.1, 2.2, 3.3], dtype=np.float32)
arr_float64 = np.array([1.1, 2.2, 3.3], dtype=np.float64)

print("Memory usage of arr_float32:", arr_float32.itemsize * arr_float32.size, "bytes")  # Output: 12 bytes (3 elements * 4 bytes/element)
print("Memory usage of arr_float64:", arr_float64.itemsize * arr_float64.size, "bytes")  # Output: 24 bytes (3 elements * 8 bytes/element)

**Example differentiating float32 and float64:**


# Create a large array with float32 and float64 data types
large_array_float32 = np.arange(1000000, dtype=np.float32)
large_array_float64 = np.arange(1000000, dtype=np.float64)

# Calculate the sum of elements
sum_float32 = np.sum(large_array_float32)
sum_float64 = np.sum(large_array_float64)

print("Sum using float32:", sum_float32)
print("Sum using float64:", sum_float64)



---



# np.zeros, np.ones, np.full:

**np.zeros:**

`np.zeros` creates an array filled with zeros. It takes the shape of the desired array as input and returns an array of that shape filled with zeros.

Syntax:
```python
numpy.zeros(shape, dtype=float)
```

- `shape`: The shape of the array (tuple of integers).
- `dtype`: Optional. The data type of the array. Default is `float`.

# Create a 2x3 array filled with zeros
zeros_array = np.zeros((2, 3))
print(zeros_array)

**np.ones:**

`np.ones` creates an array filled with ones. Similar to `np.zeros`, it takes the shape of the desired array as input and returns an array of that shape filled with ones.

Syntax:
```python
numpy.ones(shape, dtype=None)
```

- `shape`: The shape of the array (tuple of integers).
- `dtype`: Optional. The data type of the array. If not specified, the default is determined by the data type of `1`.

# Create a 3x2 array filled with ones
ones_array = np.ones((3, 2))
print(ones_array)

**np.full:**

`np.full` creates an array filled with a specified constant value. It takes the shape of the desired array and the constant value as input and returns an array of that shape filled with the specified value.

Syntax:
```python
numpy.full(shape, fill_value, dtype=None)
```

- `shape`: The shape of the array (tuple of integers).
- `fill_value`: The constant value to fill the array with.
- `dtype`: Optional. The data type of the array. If not specified, the default is determined by the data type of `fill_value`.


# Create a 2x2 array filled with 5
full_array = np.full((2, 2), 4.5)
print(full_array)



---



# Array Operations - NumPy

**1. Arithmetic Operations:**

# Addition
arr_sum = np.add([1, 2, 3], [4, 5, 6])
print("Addition:", arr_sum)  # Output: [5 7 9]

# Subtraction
arr_diff = np.subtract([5, 6, 7], [2, 3, 1])
print("Subtraction:", arr_diff)  # Output: [3 3 6]

# Multiplication
arr_prod = np.multiply([2, 3, 4], [3, 4, 5])
print("Multiplication:", arr_prod)  # Output: [ 6 12 20]

# Division
arr_div = np.divide([10, 12, 14], [2, 3, 2])
print("Division:", arr_div)  # Output: [5. 4. 7.]

# Modulus
arr_mod = np.mod([10, 11, 12], [3, 4, 5])
print("Modulus:", arr_mod)  # Output: [1 3 2]

# Exponentiation
arr_pow = np.power([2, 3, 4], [2, 3, 2])
print("Exponentiation:", arr_pow)  # Output: [ 4 27 16]


**2. Relational Operations:**

# Create sample arrays
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([2, 2, 4, 3])

# Equal
print("Equal:", arr1 == arr2)  # Output: [False  True False False]

# Not Equal
print("Not Equal:", arr1 != arr2)  # Output: [ True False  True  True]

# Greater Than
print("Greater Than:", arr1 > arr2)  # Output: [False False False  True]

# Greater Than or Equal To
print("Greater Than or Equal To:", arr1 >= arr2)  # Output: [False  True False  True]

# Less Than
print("Less Than:", arr1 < arr2)  # Output: [ True False False False]

# Less Than or Equal To
print("Less Than or Equal To:", arr1 <= arr2)  # Output: [ True  True  True False]




---



# Indexing and Slicing for 1D Arrays

**1. Indexing:**

Indexing refers to accessing individual elements of an array using their position (index) within the array. In NumPy, indexing starts from 0, so the first element has index 0, the second element has index 1, and so on.

# Create a 1D array
arr = np.array([1, 2, 3, 4, 5])

# Access individual elements using indexing
print("First element:", arr[0])   # Output: 1
print("Second element:", arr[1])  # Output: 2
print("Last element:", arr[-1])   # Output: 5 (negative indexing)

**2. Slicing:**

Slicing allows you to extract a subset of elements from an array by specifying a range of indices. The basic syntax for slicing is `start:stop:step`, where `start` is the starting index (inclusive), `stop` is the ending index (exclusive), and `step` is the step size.

# Create a 1D array
arr = np.array([1, 2, 3, 4, 5])

# Slice elements from index 1 to index 3 (exclusive)
print("Slice:", arr[1:3])  # Output: [2 3]

# Slice elements from index 0 to index 4 with step size 2
print("Slice with step:", arr[0:4:2])  # Output: [1 3]

**3. Negative Indexing:**

Negative indexing allows you to access elements from the end of the array by specifying negative indices. `-1` refers to the last element, `-2` refers to the second last element, and so on.

# Create a 1D array
arr = np.array([1, 2, 3, 4, 5])

# Access the last element using negative indexing
print("Last element:", arr[-1])  # Output: 5

**4. Slicing with Omitted Indices:**

You can omit any of the slicing parameters to use default values. Omitting `start` defaults to 0, omitting `stop` defaults to the end of the array, and omitting `step` defaults to 1.


# Create a 1D array
arr = np.array([1, 2, 3, 4, 5])

# Slice elements from the beginning to index 3 (exclusive)
print("Slice with omitted start:", arr[:3])  # Output: [1 2 3]

# Slice elements from index 2 to the end
print("Slice with omitted stop:", arr[2:])  # Output: [3 4 5]

# Slice elements with step size 2
print("Slice with omitted step:", arr[::2])  # Output: [1 3 5]



---




# Indexing and Slicing for 2D Arrays

**1. Indexing:**

Indexing refers to accessing individual elements of an array using their position (index) within the array. In a 2D array, indexing is done using row and column indices.

# Create a 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Access individual elements using indexing
print("Element at (0, 0):", arr[0, 0])  # Output: 1
print("Element at (1, 2):", arr[1, 2])  # Output: 6

**2. Slicing:**

Slicing allows you to extract a subset of elements from an array by specifying ranges of row and column indices. The basic syntax for slicing is `start:stop:step`, where `start` is the starting index (inclusive), `stop` is the ending index (exclusive), and `step` is the step size.

# Create a 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Slice elements from rows 0 to 1 (exclusive) and columns 1 to 2 (exclusive)
print("Slice:", arr[0:2, 1:3])


# Modify slice
arr[0:2, 1:3] = [[10, 20], [30, 40]]
print("Modified array after slicing:", arr)


**3. Negative Indexing:**

Negative indexing can also be used in 2D arrays to access elements from the end of the array.

# Create a 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Access the last element using negative indexing
print("Last element:", arr[-1, -1])  # Output: 9

**4. Slicing with Omitted Indices:**

You can omit any of the slicing parameters to use default values. Omitting `start` defaults to 0, omitting `stop` defaults to the end of the array, and omitting `step` defaults to 1.

# Create a 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Slice elements from rows 1 to the end and all columns
print("Slice with omitted start and stop:", arr[1:])

# Slice elements from all rows and columns 0 to 1 (exclusive) with step size 2
print("Slice with omitted step:", arr[:, 0:2:2])



---



# Mass Level Indexing and Slicing

**1. Boolean Indexing:**

Boolean indexing allows you to select elements from an array based on a condition. You create a boolean mask indicating which elements satisfy the condition, and then use this mask to extract the desired elements.

# Create a 1D array
arr_1d = np.array([1, 2, 3, 4, 5])

# Boolean mask for elements greater than 3
mask = arr_1d > 3

# Use boolean mask to select elements
selected_elements = arr_1d[mask]
print("Selected elements:", selected_elements)

# Create a 2D array
arr_2d = np.array([[1, 2, 3], # False, False, True
                    [4, 5, 6],
                    [7, 8, 9]])

# Boolean indexing to select elements greater than 2
result_2d = arr_2d[arr_2d > 2]
print("Elements greater than 2 in 2D array:", result_2d)


**2. Fancy Indexing:**

Fancy indexing allows you to select elements from an array using arrays of indices. You provide arrays of indices along each axis, and the elements at those indices are returned as a new array.

# Create a 1D array
arr_1d = np.array([1, 2, 3, 4, 5])

# Fancy indexing
indices = [0, 2, 4]
selected_elements = arr_1d[indices]
print("Selected elements:", selected_elements)

# Playing with Arrays

**1. Transposing Arrays:**

Transposing an array means exchanging its rows and columns. In NumPy, you can transpose an array using the `T` attribute or the `transpose()` function.

# Create a 2D array
arr_2d = np.array([[1, 2, 3],
                    [4, 5, 6]])

# Transpose the array
transposed_arr = arr_2d.T
print("Transposed array:")
print(transposed_arr)

**2. Swapping Axes:**

Swapping axes means rearranging the dimensions of an array. You can swap axes using the `swapaxes()` function.

# Create a 2D array
arr_2d = np.array([[1, 2, 3],
                    [4, 5, 6]])

# Swap axes
swapped_arr = arr_2d.swapaxes(0,1)
print("Swapped array:")
print(swapped_arr)

# Create a 3D array of shape (2, 3, 4)
# Think of it as 2 layers of 3x4 matrices
array_3d = np.array([[[ 1,  2,  3,  4],
                      [ 5,  6,  7,  8],
                      [ 9, 10, 11, 12]],

                     [[13, 14, 15, 16],
                      [17, 18, 19, 20],
                      [21, 22, 23, 24]]])

print("Original array shape:", array_3d.shape)

# Swap the first and last axes (depth with columns)
swapped_array = np.swapaxes(array_3d, 0, 2)

print("Swapped array shape:", swapped_array.shape)
print("Swapped array data:\n", swapped_array)


**3. Pseudo-random Number Generation:**

NumPy provides various functions for generating pseudo-random numbers. These functions are located in the `numpy.random` module. You can generate random numbers from different distributions, such as uniform, normal, binomial, etc.

# Pseudo-random Number Generation in 1D Array:

# Generate 5 random integers between 1 and 10
random_integers = np.random.randint(1, 10, size=5)
print("Random integers (1D):", random_integers)

# Generate 5 random numbers from a normal distribution
random_normal = np.random.normal(size=5)
print("Random numbers from normal distribution (1D):", random_normal)

# Pseudo-random Number Generation in 2D Array:

# Generate a 2D array of shape (3, 3) with random integers between 1 and 10
random_integers_2d = np.random.randint(1, 10, size=(3, 3))
print("Random integers (2D):")
print(random_integers_2d)

# Generate a 2D array of shape (3, 3) with random numbers from a normal distribution
random_normal_2d = np.random.normal(size=(3, 3))
print("Random numbers from normal distribution (2D):")
print(random_normal_2d)



---



# Masking

Masking in NumPy involves using boolean arrays (masks) to filter or select elements from arrays based on certain conditions. This is particularly useful for selecting elements that satisfy specific criteria.

**1. Masking in 1D Array:**

# Create a 1D array
arr_1d = np.array([1, 2, 3, 4, 5])

# Create a boolean mask based on a condition (e.g., elements greater than 2)
mask_1d = arr_1d > 2

# Apply the mask to select elements from the array
result_1d = arr_1d[mask_1d]

print("Original 1D array:", arr_1d)
print("Boolean mask:", mask_1d)
print("Selected elements using mask:", result_1d)

**2. Masking in 2D Array:**

# Create a 2D array
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Create a boolean mask based on a condition (e.g., elements greater than 5)
mask_2d = arr_2d > 5

# Apply the mask to select elements from the array
result_2d = arr_2d[mask_2d]

print("Original 2D array:")
print(arr_2d)

print("Boolean mask:")
print(mask_2d)

print("Selected elements using mask:")
print(result_2d)



---



# Operations on 2D Arrays

**1. Matrix Multiplication (`np.matmul()`):**

Matrix multiplication is a fundamental operation in linear algebra, where you multiply two matrices to obtain a new matrix. In NumPy, you can perform matrix multiplication using the `np.matmul()` function.

# Define matrices
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

# Matrix multiplication using np.matmul()
result = np.matmul(matrix_a, matrix_b)
print("Matrix Multiplication:")
print(result)

**2. Reshaping (`np.reshape()`):**

Reshaping an array means changing the shape of the array without changing its data. It's useful for converting arrays between different dimensions or rearranging their layout.

# Reshaping an array
arr = np.arange(1, 10)  # 1D array from 1 to 9
print(arr)
reshaped_arr = arr.reshape((3, 3))  # Reshape to a 3x3 matrix
print("Reshaped array:")
print(reshaped_arr)

**3. Transpose (`np.transpose()`):**

Transposing a matrix means flipping its rows with its columns. In NumPy, you can obtain the transpose of a matrix using the `np.transpose()` function or the `.T` attribute.

# Transposing a matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
transposed_matrix = np.transpose(matrix)
print("Transposed matrix:")
print(transposed_matrix)

**4. Aggregate Functions:**

Aggregate functions in NumPy are functions that operate on arrays and return a single value, summarizing the data in some way. Common aggregate functions include `np.sum()`, `np.max()`, `np.min()`, `np.mean()`, etc.

# Aggregate functions
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print("Sum of all elements:", np.sum(matrix))  # Output: 21
print("Maximum element:", np.max(matrix))  # Output: 6
print("Minimum element:", np.min(matrix))  # Output: 1
print("Mean of all elements:", np.mean(matrix))  # Output: 3.5



---



# Universal Functions (ufuncs)

**1. Basic Arithmetic Operations:**

# Create a sample array
arr = np.array([1, 2, 3, 4, 5])

# Element-wise addition
result_add = np.add(arr, 2)  # Add 2 to each element
print("Addition:", result_add)

# Element-wise multiplication
result_mul = np.multiply(arr, 3)  # Multiply each element by 3
print("Multiplication:", result_mul)

**2. Trigonometric Functions:**

# Trigonometric functions
angles = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])

# Sine
result_sin = np.sin(angles)
print("Sine:", result_sin)

# Cosine
result_cos = np.cos(angles)
print("Cosine:", result_cos)

**3. Exponential and Logarithmic Functions:**

# Exponential and logarithmic functions
arr = np.array([1, 2, 3, 4, 5])

# Exponential
result_exp = np.exp(arr)
print("Exponential:", result_exp)

# Natural logarithm
result_log = np.log(arr)
print("Natural Logarithm:", result_log)

**4. Statistical Functions:**

# Statistical functions
arr = np.array([1, 2, 3, 4, 5])

# Mean
result_mean = np.mean(arr)
print("Mean:", result_mean)

# Standard deviation
result_std = np.std(arr)
print("Standard Deviation:", result_std)

**5. Comparison Functions:**

# Comparison functions
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([2, 3, 3, 4, 4])

# Greater than
result_gt = np.greater(arr1, arr2)
print("Greater Than:", result_gt)

# Less than or equal to
result_lte = np.less_equal(arr1, arr2)
print("Less Than or Equal To:", result_lte)

**6. Broadcasting:**

Ufuncs also support broadcasting, which means they can operate on arrays of different shapes. NumPy automatically broadcasts arrays to perform element-wise operations.

arr = np.array([[1, 2, 3], [4, 5, 6]])

# Element-wise addition with scalar
result_broadcast = arr + 2
print("Broadcasting with Scalar:")
print(result_broadcast)



---



# Array Manipulations

# Playing with Shapes

**1. reshape:** The reshape method returns a new array with the specified shape, without changing the data.

# Create a one-dimensional array of 12 elements
a = np.arange(12)
print("Original array:", a)

# Reshape it to a 3x4 two-dimensional array
b = a.reshape(3,4)
print("Reshaped array:\n", b)

**2. resize:** The resize method changes the shape and size of an array in-place. This method can alter the original array and fill in with repeated copies of a if the new array is larger than the original.

# Resize the array in-place to 2x6
a = np.arange(10)
a.resize(2, 6)
print("Resized array:\n", a)

**3. ravel:** The ravel method returns a flattened one-dimensional array. It's a convenient way to convert any multi-dimensional array into a flat 1D array.

# Flatten the 3x4 array to a one-dimensional array
print(b)
flat = b.ravel()
print("Flattened array:", flat)

**4. flatten:** Similar to ravel, but flatten returns a copy instead of a view of the original data, thus not affecting the original array.

# Create a copy of flattened array
print(b)
flat_copy = b.flatten()
print("Flattened array copy:", flat_copy)

**Difference between ravel and flatten:**

# Creating a 2D array
a = np.array([[1, 2], [3, 4]])

# Flattening using ravel
b = a.ravel()
b[0] = 100  # Modifying the raveled array

# Flattening using flatten
c = a.flatten()
c[1] = 200  # Modifying the flattened array

print("Original array after modifying raveled array:", a)
print("Original array after modifying flattened array does not change:", a)


**5. squeeze:** The squeeze method is used to remove axes of length one from an array.

# Create an array with a singleton dimension
c = np.array([[[1, 2, 3, 4]]])
print("Original array with singleton dimension:", c.shape)

# Squeeze to remove singleton dimensions
squeezed = c.squeeze()
print(squeezed)
print("Squeezed array:", squeezed.shape)

**6. expand_dims:** The opposite of squeeze, expand_dims is used to add an axis at a specified position.

# Add an axis at index 1
expanded = np.expand_dims(squeezed, axis=1)
print(expanded)
print("Expanded array shape:", expanded.shape)

# Splitting and Joining
Splitting allows you to divide large arrays into smaller arrays. This can be useful for parallel processing tasks or during situations where subsets of data need to be analyzed separately.

**1. np.split:** Splits an array into multiple sub-arrays.

x = np.arange(9)
print("Original array:", x)

# Split the array into 3 equal parts
x_split = np.split(x, 3)
print("Split array:", x_split)

**2. np.array_split:** Similar to np.split, but allows for splitting into unequal subarrays.

# Split the array into 4 parts, which will not be equal
x_array_split = np.array_split(x, 4)
print("Array split into unequal parts:", x_array_split)

**3. np.hsplit and np.vsplit:** These are specific cases of split for horizontal and vertical splitting respectively, useful for 2D arrays (matrices).

y = np.array([[1, 2, 3], [4, 5, 6]])
print("Original 2D array:\n", y)

# Horizontal split
y_hsplit = np.hsplit(y, 3)
print("Horizontally split:", y_hsplit)

# Vertical split
y_vsplit = np.vsplit(y, 2)
print("Vertically split:", y_vsplit)

**4. np.concatenate:** Concatenates a sequence of arrays along an existing axis.

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Concatenate along the first axis
concatenated = np.concatenate((a, b))
print("Concatenated array:", concatenated)

**5. np.hstack and np.vstack:** These are specific cases of concatenate for horizontal and vertical stacking respectively.

# Horizontal stack
h_stacked = np.hstack((a, b))
print("Horizontally stacked:", h_stacked)

# Vertical stack
v_stacked = np.vstack((a, b))
print("Vertically stacked:\n", v_stacked)

# Adding and Removing Elements
These operations allow you to modify array sizes dynamically.

**1. np.append:** Adds elements to the end of an array.

# Append elements to the array
a = np.array([1, 2, 3])
appended = np.append(a, [7, 8])
print("Appended array:", appended)

**2. np.insert:** Inserts elements at a specific position in the array.

# Insert elements into the array
inserted = np.insert(a, 1, [9, 10])
print("Array with inserted elements:", inserted)

**3. np.delete:** Removes elements at a specific position from the array.

# Create a one-dimensional array
a = np.array([1, 2, 3, 4, 5])

# Delete the element at index 2
result = np.delete(a, 2)
print("Array after deleting element at index 2:", result)

# Delete multiple elements
result = np.delete(a, [0, 3])
print("Array after deleting elements at indices 0 and 3:", result)

# Create a two-dimensional array
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Delete the second row
result = np.delete(b, 1, axis=0)
print("Array after deleting second row:\n", result)

# Delete the third column
result = np.delete(b, 2, axis=1)
print("Array after deleting third column:\n", result)
