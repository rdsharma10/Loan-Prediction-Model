"""
NumPy Cheatsheet
Author: Rishi Sharma
Description: This file contains a comprehensive NumPy cheatsheet with essential functions and operations.
"""
import numpy as np

# Creating Arrays
arr1 = np.array([1, 2, 3, 4, 5])  # 1D array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array
arr_zeros = np.zeros((3, 3))  # 3x3 matrix of zeros
arr_ones = np.ones((3, 3))  # 3x3 matrix of ones
arr_identity = np.eye(3)  # 3x3 identity matrix
arr_random = np.random.rand(3, 3)  # 3x3 matrix with random values
arr_arange = np.arange(0, 10, 2)  # Array from 0 to 10 with step 2
arr_linspace = np.linspace(0, 1, 5)  # 5 evenly spaced values from 0 to 1

# Array Properties
shape = arr2.shape  # Shape of the array
size = arr2.size  # Total number of elements
dtype = arr2.dtype  # Data type of elements
ndim = arr2.ndim  # Number of dimensions

# Reshaping and Flattening
reshaped = arr2.reshape((3, 2))  # Reshape array
flattened = arr2.flatten()  # Flatten array

# Basic Operations
sum_all = arr1.sum()  # Sum of all elements
mean_value = arr1.mean()  # Mean of elements
max_value = arr1.max()  # Maximum value
min_value = arr1.min()  # Minimum value
std_dev = arr1.std()  # Standard deviation

# Element-wise Operations
arr_add = arr1 + 2  # Add scalar to array
arr_mul = arr1 * 2  # Multiply array by scalar
arr_sqrt = np.sqrt(arr1)  # Square root
arr_exp = np.exp(arr1)  # Exponential function
arr_log = np.log(arr1 + 1)  # Logarithm (avoid log(0))

# Matrix Operations
matrix_mult = np.dot(arr2, arr2.T)  # Matrix multiplication
transpose = arr2.T  # Transpose of matrix
inv_matrix = np.linalg.inv(np.array([[2, 3], [1, 4]]))  # Inverse of matrix
det_matrix = np.linalg.det(np.array([[2, 3], [1, 4]]))  # Determinant

# Indexing and Slicing
first_element = arr1[0]  # First element
slice_part = arr1[1:4]  # Slicing elements 1 to 4
row = arr2[1, :]  # Second row of 2D array
column = arr2[:, 1]  # Second column of 2D array

# Conditional Selection
filtered = arr1[arr1 > 2]  # Select elements greater than 2

# Stacking Arrays
vert_stack = np.vstack((arr1, arr1))  # Vertical stack
horz_stack = np.hstack((arr1, arr1))  # Horizontal stack

# Saving and Loading Arrays
np.save("array.npy", arr1)  # Save array to file
loaded_array = np.load("array.npy")  # Load array from file
