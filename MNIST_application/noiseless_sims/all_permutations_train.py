'''
Code to perform a sweep on all possible permutations of input configurations (3 qubits only -> 6!=720 combinations)
'''

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
import numpy as np
import itertools
from utils import set_seed
import os

set_seed(123)

# Define the hyperparameters
batch_size = 10

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

# Filter the training data for only classes 0 and 1
train_mask = (train_labels == cls1) | (train_labels == cls2)
filtered_train_data = compressed_train[train_mask]
filtered_train_labels = train_labels[train_mask]

# Filter the testing data for only classes 0 and 1
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
n_qubits = 3
n_classes = 2
n_layers = 3
n_features = 6

# Create a quantum and classical device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = qml.device("default.qubit", wires=n_qubits)

indices = list(range(n_features))

tr_acc_list = []
inf_acc_list = []

for case, ind_list in enumerate(list(itertools.permutations(indices))):
    print("**************************************")
    print(f"Case {case+1}: {ind_list}")
    print("**************************************")

    set_seed(123)

    # Define the quantum circuit
    @qml.qnode(dev, interface='torch')
    def qcircuit(inputs, params):
        qml.RX(inputs[:,ind_list[0]],wires=0)
        qml.RY(inputs[:,ind_list[1]],wires=0)
        qml.RX(inputs[:,ind_list[2]],wires=1)
        qml.RY(inputs[:,ind_list[3]],wires=1)
        qml.RX(inputs[:,ind_list[4]],wires=2)
        qml.RY(inputs[:,ind_list[5]],wires=2)
        qml.templates.StronglyEntanglingLayers(params, wires=range(n_qubits), ranges=[1,1,1])
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_classes)]

    # Define the quantum model
    weight_shapes = {"params": (n_layers, n_qubits, 3)}

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, init_method=torch.nn.init.normal_).cuda()

        def forward(self, x):
            out = self.qlayer(x)
            return out
    
    model = Model().to(device)

    # Define the loss function, epochs, optimizer
    epochs = 5
    loss_fn = nn.CrossEntropyLoss()
    soft_out = nn.Softmax(dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-3)

    # Define the training loop
    for epoch in range(epochs):
        train_acc = test_acc = 0
        loss_list = []
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = soft_out(model(inputs))
            pred = torch.argmax(outputs, dim=1)
            loss = loss_fn(outputs, labels.long())
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            dist = torch.abs(labels - pred)
            train_acc += len(dist[dist == 0])
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = soft_out(model(inputs))
            pred = torch.argmax(outputs, dim=1)
            dist = torch.abs(labels - pred)
            test_acc += len(dist[dist == 0])

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Train Accuracy: {train_acc/len(filtered_train_dataset)*100:.2f}, Test Accuracy: {test_acc/len(filtered_test_dataset)*100:.2f}")
        
        # Save final train and inferencing accuracies
        if epoch == epochs - 1:
            tr_acc_list.append(train_acc/len(filtered_train_dataset)*100)
            inf_acc_list.append(test_acc/len(filtered_test_dataset)*100)
    
    if not os.path.exists(f"brute_force_{cls1}_{cls2}.txt"):
        with open(f"brute_force_{cls1}_{cls2}.txt", 'w+') as f:
            lines = f.readlines()
    else: 
        with open(f"brute_force_{cls1}_{cls2}.txt", 'r') as f:
            lines = f.readlines()
    if case + 1 == 1: # First permutation
        if (len(lines) < 2) or (lines[0] != "" and lines[1] != ""):
            lines.insert(0, "")
            lines.insert(0, "")
    if case + 1 < len(list(itertools.permutations(indices))): # First P-1 permutations
        lines[0] = lines[0].strip() + str(tr_acc_list[-1])+',\n'
        lines[1] = lines[1].strip() + str(inf_acc_list[-1])+',\n'
    else: # Final permutation
        lines[0] = lines[0].strip() + str(tr_acc_list[-1])+'\n'
        lines[1] = lines[1].strip() + str(inf_acc_list[-1])+'\n'
    with open(f"brute_force_{cls1}_{cls2}.txt", 'w') as f: # Write lines
        f.writelines(lines)

# Print results for all index cases
print(f"Training accuracy values: {tr_acc_list}")
print(f"Inferencing accuracy values: {inf_acc_list}")

