'''
Code to train tiny-imagenet dataset using the compressed data
'''

import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from utils import set_seed
import pennylane as qml
import torch.nn as nn
import time
from functools import partial
from utils import * 

set_seed(123)

# Load the compressed data from .npy files. Choose either 'reduced_images.npy' or 'reduced_images_hi.npy'
# reduced_images.npy: bell pepper - orange
# reduced_images_hi.npy: sulphur butterfly - alp
compressed_imagenet = np.load('reduced_images_hi.npy')
labels = np.load('labels_hi.npy')

# Convert labels to 0 and 1
unique_labels = np.unique(labels)
label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
labels = np.vectorize(label_map.get)(labels)

# Shuffle the dataset
permutation = np.random.permutation(len(compressed_imagenet))
compressed_imagenet = compressed_imagenet[permutation]
labels = labels[permutation]

# Split to train and test
compressed_train = compressed_imagenet[:800]
compressed_test = compressed_imagenet[800:]
train_labels = labels[:800]
test_labels = labels[800:]

compressed_train = (compressed_train - compressed_train.mean(axis=0)) / compressed_train.std(axis=0)
compressed_test = (compressed_test - compressed_test.mean(axis=0)) / compressed_test.std(axis=0)

# Print shapes to verify
print("Loaded compressed training set shape:", compressed_train.shape)
print("Loaded compressed test set shape:", compressed_test.shape)

# Convert the data and labels to torch tensors
filtered_train_data_tensor = torch.tensor(compressed_train, dtype=torch.float32)
filtered_train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

filtered_test_data_tensor = torch.tensor(compressed_test, dtype=torch.float32)
filtered_test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# Create TensorDatasets
filtered_train_dataset = TensorDataset(filtered_train_data_tensor, filtered_train_labels_tensor)
filtered_test_dataset = TensorDataset(filtered_test_data_tensor, filtered_test_labels_tensor)

# Create DataLoaders
train_loader = DataLoader(dataset=filtered_train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=filtered_test_dataset, batch_size=10, shuffle=False)

####################################################################################

# Define quantum parameters
n_qubits = 8
n_classes = 2
n_layers = 3
range_param = 1

# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Use a device that supports parameter broadcasting
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum circuit with parameter broadcasting
start = time.time()
for _ in range(10):
    @qml.qnode(dev, interface='torch')
    def qcircuit(inputs, params):
        custom_rx(inputs[:,0],wires=0)
        custom_ry(inputs[:,1],wires=0)
        custom_rx(inputs[:,2],wires=1)
        custom_ry(inputs[:,3],wires=1)
        custom_rx(inputs[:,4],wires=2)
        custom_ry(inputs[:,5],wires=2)
        if n_qubits in list(range(4, 9)):
            custom_rx(inputs[:,6],wires=3)
            custom_ry(inputs[:,7],wires=3)
        if n_qubits in list(range(5, 9)):
            custom_rx(inputs[:,8],wires=4)
            custom_ry(inputs[:,9],wires=4)
        if n_qubits in list(range(6, 9)):
            custom_rx(inputs[:,10],wires=5)
            custom_ry(inputs[:,11],wires=5)
        if n_qubits in list(range(7, 9)):
            custom_rx(inputs[:,12],wires=6)
            custom_ry(inputs[:,13],wires=6)
        if n_qubits == 8:
            custom_rx(inputs[:,14],wires=7)
            custom_ry(inputs[:,15],wires=7)
        for l in range(n_layers):
            # Rotations
            for q in range(n_qubits):
                phi = params[l, q, 0]
                theta = params[l, q, 1]
                omega = params[l, q, 2]
                custom_rot(phi, theta, omega, wires=q)
            # Entangling gates
            for q in range(n_qubits):
                target = (q + range_param) % n_qubits
                custom_CNOT(wires=[q, target])
        # Apply readout errors before measurement
        readout_error(range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_classes)]

    weight_shapes = {"params": (n_layers, n_qubits, 3)}
    layout_coupling = [(i,i+1) for i in range(n_qubits-1)]

    # Insert SWAPs
    transpiled_ckt = qml.transforms.transpile(qcircuit, layout_coupling)
    qcircuit = qml.QNode(transpiled_ckt, dev)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            # No need to send qlayer to CUDA separately; the whole model will be sent to device
            self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes)
            self.relu = nn.ReLU()
            self.lin = nn.Linear(n_classes, n_classes)

        def forward(self, x):
            # x shape: [batch_size, n_features]
            out = self.qlayer(x)
            return out

    model = Model().to(device)

    # Define the loss function, epochs, optimizer
    epochs = 5
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    soft_out = nn.Softmax(dim=1)

    # Define the training loop
    for epoch in range(epochs):
        model.train()
        train_acc = 0
        loss_list = []
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = soft_out(outputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            pred = torch.argmax(outputs, dim=1)
            train_acc += (pred == labels).sum().item()
        model.eval()
        test_acc = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = soft_out(outputs)
                pred = torch.argmax(outputs, dim=1)
                test_acc += (pred == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(loss_list):.4f}, Train Accuracy: {train_acc/len(filtered_train_dataset)*100:.2f}, Test Accuracy: {test_acc/len(filtered_test_dataset)*100:.2f}")

end = time.time()
elapsed_time = end - start
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Execution time:{int(hours)} hrs, {int(minutes)} mins, {seconds:.2f} secs")
