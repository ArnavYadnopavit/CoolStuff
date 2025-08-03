import ctypes
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Load the shared library
lib = ctypes.CDLL('./svm.so')

# Set argument and return types for the function
lib.FindHyperplane.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_double]
lib.FindHyperplane.restype = ctypes.POINTER(ctypes.c_double)

DIM = 2

def call_find_hyperplane(x, y):
    # Convert Python list to ctypes array
    x_arr = (ctypes.c_double * DIM)(*x)
    y_val = ctypes.c_double(y)

    # Call the C function
    result_ptr = lib.FindHyperplane(x_arr, y_val)

    # Convert result to Python list
    result = [result_ptr[i] for i in range(DIM)]

    return result

# --- Generate data ---
np.random.seed(0)
num_points = 100
X = np.random.randn(num_points, 2)
true_w = np.array([2, -1])
b = 0.5
Y = np.sign(X @ true_w + b)

# Apply FindHyperplane to each (x, y)
transformed = [call_find_hyperplane(x, y) for x, y in zip(X, Y)]
transformed = np.array(transformed)

# --- Plotting ---
plt.figure(figsize=(8, 6))

for i in range(num_points):
    if Y[i] == 1:
        plt.scatter(X[i, 0], X[i, 1], c='blue', label='Positive' if i == 0 else "")
    else:
        plt.scatter(X[i, 0], X[i, 1], c='red', label='Negative' if i == 0 else "")

# Plot the true hyperplane: 2x - y + 0.5 = 0 → y = 2x + 0.5
x_vals = np.linspace(-3, 3, 100)
y_vals = 2 * x_vals + 0.5
plt.plot(x_vals, y_vals, 'k--', label='True Hyperplane')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Data and Hyperplane with C Function Output (via ctypes)')
plt.grid(True)
plt.show()

