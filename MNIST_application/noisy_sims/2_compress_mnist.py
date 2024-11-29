'''Compress MNIST dataset using PCA and save the compressed dataset.'''

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from utils import set_seed
from sklearn.decomposition import PCA
import os

set_seed(123)

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

if not os.path.exists("./mnist_data"):
    os.mkdir("./mnist_data")

# Load the MNIST training and testing datasets
train_dataset = torchvision.datasets.MNIST(
    root='./mnist_data', train=True, transform=transform, download=False
)
test_dataset = torchvision.datasets.MNIST(
    root='./mnist_data', train=False, transform=transform, download=False
)

# Define DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Compress training and test sets
train_data = train_dataset.data.numpy().reshape(len(train_dataset), -1)
test_data = test_dataset.data.numpy().reshape(len(test_dataset), -1)
pca = PCA(n_components=16)

# Using PCA
compressed_train = pca.fit_transform(train_data)
compressed_test = pca.transform(test_data)

# Obtain train and test labels
train_labels = train_dataset.targets.numpy()
test_labels = test_dataset.targets.numpy()

# Make directory for compressed data if it doesn't exist
if not os.path.exists("./compressed_data"):
    os.mkdir("./compressed_data")

# Save compressed datasets to files
np.save('./compressed_data/compressed_mnist_train.npy', compressed_train)
np.save('./compressed_data/compressed_mnist_test.npy', compressed_test)
np.save('./compressed_data/train_labels.npy', train_labels)
np.save('./compressed_data/test_labels.npy', test_labels)

print("Compressed datasets saved successfully!")
