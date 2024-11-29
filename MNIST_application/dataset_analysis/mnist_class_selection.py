import os
import torchvision
from torchvision import datasets, transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from itertools import combinations
import warnings
from sklearn.metrics import silhouette_score
from utils import set_seed

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

set_seed(123)

# Load the compressed data from .npy files
compressed_train = np.load('compressed_mnist_train.npy')
compressed_test = np.load('compressed_mnist_test.npy')

# Normalize the compressed data
# compressed_train = (compressed_train - compressed_train.mean(axis=0)) / compressed_train.std(axis=0)
# compressed_test = (compressed_test - compressed_test.mean(axis=0)) / compressed_test.std(axis=0)

# Print shapes to verify
print("Loaded compressed training set shape:", compressed_train.shape)
print("Loaded compressed test set shape:", compressed_test.shape)

# Load the original MNIST dataset to get the labels
train_dataset = datasets.MNIST(
    root='./mnist_data', train=True, download=False
)
test_dataset = datasets.MNIST(
    root='./mnist_data', train=False, download=False
)

# Get the original labels for train and test sets
train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')

# Classes in MNIST (digits 0 through 9)
classes = list(set(train_labels))

# Get all possible class pairs (combinations of two classes)
class_pairs = list(combinations(classes, 2))

silhouette_scores = []

# Loop through each pair of classes and compute Silhouette Coefficient
for pair in class_pairs:
    first_two_classes = pair
    print(f"Processing classes: {first_two_classes}")
    
    # Filter the compressed training data and labels for the selected class pair
    class_mask = (train_labels == first_two_classes[0]) | (train_labels == first_two_classes[1])
    filtered_train_data = compressed_train[class_mask][:800]
    filtered_train_labels = train_labels[class_mask][:800]
    class_mask = (test_labels == first_two_classes[0]) | (test_labels == first_two_classes[1])
    filtered_test_data = compressed_test[class_mask][:200]
    filtered_test_labels = test_labels[class_mask][:200]

    filtered_data = np.concatenate((filtered_train_data, filtered_test_data), axis=0)
    filtered_labels = np.concatenate((filtered_train_labels, filtered_test_labels), axis=0)
    
    # Compute the silhouette score for the reduced data
    ss = silhouette_score(filtered_data, filtered_labels)
    silhouette_scores.append(ss)
    
    # Log the silhouette score for the class pair
    print(f"Separation score for classes {first_two_classes}: {ss}")
    with open('class_selection_log.txt', 'a') as f:
        f.write(f"Classes: {first_two_classes}, Separation Score: {ss}\n")
