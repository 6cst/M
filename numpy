import numpy as np

arr1 = np.array([1, 2, 3, 4, 5])
print("1D Array:", arr1)

arr2 = np.array([[1, 2], [3, 4]])
print("2D Array:\n", arr2)

zeros_array = np.zeros((2, 3))
print("Zeros Array:\n", zeros_array)

ones_array = np.ones((3, 3))
print("Ones Array:\n", ones_array)

arange_array = np.arange(0, 10, 2)
print("Arange Array:", arange_array)

linspace_array = np.linspace(0, 1, 5)
print("Linspace Array:", linspace_array)

arr = np.array([10, 20, 30])
print("Addition:", arr + 5)
print("Multiplication:", arr * 2)

mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])
result = np.dot(mat1, mat2)
print("Matrix Multiplication:\n", result)

print("Element at index 1:", arr1[1])

sliced = arr1[1:4]
print("Sliced Array:", sliced)

reshaped = np.reshape(arr2, (4, 1))
print("Reshaped Array:\n", reshaped)

concatenated = np.concatenate((arr1, arange_array))
print("Concatenated Array:", concatenated)

arr = np.array([0, np.pi/2, np.pi])
print("Sin:", np.sin(arr))
print("Exponential:", np.exp(arr))
print("Mean:", np.mean(arr1))
print("Standard Deviation:", np.std(arr1))

print("Greater than 2:", arr1 > 2)

random_array = np.random.rand(3, 3)
print("Random Array:\n", random_array)

unsorted = np.array([3, 1, 2])
print("Sorted Array:", np.sort(unsorted))
