import ctypes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Constants
DIM = 2
N = 100

# Define C structs in Python
class Data(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double * DIM), ("y", ctypes.c_double)]

class SVM(ctypes.Structure):
    _fields_ = [("w", ctypes.POINTER(ctypes.c_double)), ("b", ctypes.c_double)]

# Load shared library
lib = ctypes.CDLL('./svm.so')
lib.FindHyperplane.argtypes = [ctypes.POINTER(Data), ctypes.c_int]
lib.FindHyperplane.restype = SVM

# Generate dataset
X, y = make_blobs(n_samples=N, centers=2, n_features=2, cluster_std=2, random_state=42)
y = 2 * y - 1  # Convert (0, 1) to (-1, +1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Prepare training data for C
train_data = (Data * len(X_train))()
for i in range(len(X_train)):
    train_data[i] = Data((ctypes.c_double * DIM)(*X_train[i]), y_train[i])

# Call C function to get model
model = lib.FindHyperplane(train_data, len(X_train))
w = [model.w[i] for i in range(DIM)]
b = model.b
print(f"Hyperplane from C: w = {w}, b = {b:.4f}")

# Test prediction
def predict(x):
    return np.sign(np.dot(w, x) + b)

y_pred = np.array([predict(x) for x in X_test])
test_accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Plotting helper
def plot_dataset(X, y, title, w, b):
    X_pos = X[y == 1]
    X_neg = X[y == -1]

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pos[:, 0], X_pos[:, 1], color='blue', label='Class +1')
    plt.scatter(X_neg[:, 0], X_neg[:, 1], color='red', label='Class -1')

    # Plot hyperplane
    x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    if w[1] != 0:
        y_vals = [-(w[0] * x + b) / w[1] for x in x_vals]
        plt.plot(x_vals, y_vals, 'k--', label='Hyperplane')
    else:
        plt.axvline(x=-b / w[0], color='k', linestyle='--', label='Hyperplane')

    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../Documentation/figs/{title}")
    plt.show()

# Plot training set
plot_dataset(X_train, y_train, 'TrainingDataWithHyperplane', w, b)

# Plot test set
plot_dataset(X_test, y_test, 'TestDataWithHyperplane', w, b)

