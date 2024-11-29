'''
Code for training a Support Vector Machine (SVM) with a quantum kernel using the Pennylane library.
'''

import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from utils import set_seed
import pennylane as qml
import torch.nn as nn
import time
import itertools
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

set_seed(123)

# Load the compressed data from .npy files
comp_data_dir = './compressed_data'
compressed_train = np.load(f'{comp_data_dir}/compressed_mnist_train.npy')
compressed_test = np.load(f'{comp_data_dir}/compressed_mnist_test.npy')

compressed_train = (compressed_train - compressed_train.mean(axis=0)) / compressed_train.std(axis=0)
compressed_test = (compressed_test - compressed_test.mean(axis=0)) / compressed_test.std(axis=0)

# Print shapes to verify
print("Loaded compressed training set shape:", compressed_train.shape)
print("Loaded compressed test set shape:", compressed_test.shape)

# Load the original MNIST dataset to get the labels
train_dataset = torchvision.datasets.MNIST(
    root='./mnist_data', train=True, download=False
)
test_dataset = torchvision.datasets.MNIST(
    root='./mnist_data', train=False, download=False
)

# Get the original labels for train and test sets
train_labels = train_dataset.targets.numpy()
test_labels = test_dataset.targets.numpy()

# Define binary classes for selection
cls1 = int(input("Enter first class: "))
cls2 = int(input("Enter second class: "))

# Filter the training data for only classes cls1 and cls2
train_mask = (train_labels == cls1) | (train_labels == cls2)
filtered_train_data = compressed_train[train_mask]
filtered_train_labels = train_labels[train_mask]

# Filter the testing data for only classes cls1 and cls2
test_mask = (test_labels == cls1) | (test_labels == cls2)
filtered_test_data = compressed_test[test_mask]
filtered_test_labels = test_labels[test_mask]

# Convert labels to 0 and 1
unique_labels = np.unique(filtered_train_labels)
label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
filtered_train_labels = np.vectorize(label_map.get)(filtered_train_labels)
filtered_test_labels = np.vectorize(label_map.get)(filtered_test_labels)

filtered_train_data = filtered_train_data[:800]
filtered_train_labels = filtered_train_labels[:800]
filtered_test_data = filtered_test_data[:200]
filtered_test_labels = filtered_test_labels[:200]

# Convert the filtered data and labels to torch tensors
filtered_train_data_tensor = torch.tensor(filtered_train_data, dtype=torch.float32)
filtered_train_labels_tensor = torch.tensor(filtered_train_labels, dtype=torch.long)

filtered_test_data_tensor = torch.tensor(filtered_test_data, dtype=torch.float32)
filtered_test_labels_tensor = torch.tensor(filtered_test_labels, dtype=torch.long)

# Create TensorDatasets for both training and testing sets
filtered_train_dataset = TensorDataset(filtered_train_data_tensor, filtered_train_labels_tensor)
filtered_test_dataset = TensorDataset(filtered_test_data_tensor, filtered_test_labels_tensor)

# Create DataLoaders for both training and testing sets
train_loader = DataLoader(dataset=filtered_train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=filtered_test_dataset, batch_size=10, shuffle=False)

####################################################################################

# Define quantum parameters
n_qubits = 4
n_classes = 2
n_layers = 3
n_features = n_qubits * 2

# Ensure that the data has the correct number of features
X_train = filtered_train_data_tensor.numpy()
X_test = filtered_test_data_tensor.numpy()

X_train = X_train[:, :n_qubits]
X_test = X_test[:, :n_qubits]

y_train = filtered_train_labels_tensor.numpy()
y_test = filtered_test_labels_tensor.numpy()

# Define the quantum device
dev_kernel = qml.device("lightning.qubit", wires=n_qubits)

# Define the projector
projector = np.zeros((2 ** n_qubits, 2 ** n_qubits))
projector[0, 0] = 1

# Define the quantum kernel
@qml.qnode(dev_kernel)
def kernel(x1, x2):
    """The quantum kernel."""
    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

def kernel_matrix(A, B):
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[kernel(a, b) for b in B] for a in A])

svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)

with dev_kernel.tracker:
    predictions = svm.predict(X_test)
    pred_train = svm.predict(X_train)
    print(accuracy_score(pred_train, y_train), accuracy_score(predictions, y_test))