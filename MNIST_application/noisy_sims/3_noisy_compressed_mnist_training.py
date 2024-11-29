'''Train compressed MNIST dataset on QNN with angle embedding correlated features on same qubit.'''

# Import necessary libraries
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
import torch.nn as nn
import time
import numpy.random as npr  # For randomness inside the circuit
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit import QuantumCircuit, transpile
import torch.autograd.profiler as profiler
from utils import *

# Set seed for reproducibility
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
n_qubits = int(input("Enter number of qubits: "))
n_classes = 2
n_layers = 3
range_param = 1

# Initialize the FakeBackend. Here, we use FakeBrisbane as our reference backend.
backend = FakeBrisbane()
properties = backend.properties()
configuration = backend.configuration()

# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = qml.device("default.qubit", wires=n_qubits)

start = time.time()
for _ in range(10):
    # Define the quantum circuit with noise
    @qml.qnode(dev, interface='torch', diff_method='best')
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

    # Define the quantum model
    weight_shapes = {"params": (n_layers, n_qubits, 3)}
    layout_coupling = [(i,i+1) for i in range(n_qubits-1)]

    transpiled_ckt = qml.transforms.transpile(qcircuit, layout_coupling)
    qcircuit = qml.QNode(transpiled_ckt, dev)

    # Define the quantum model
    weight_shapes = {"params": (n_layers, n_qubits, 3)}

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, init_method=torch.nn.init.normal_).to(device)

        def forward(self, x):
            out = self.qlayer(x)
            return out

    model = Model().to(device)

    # Define the loss function, epochs, optimizer
    epochs = 5
    loss_fn = nn.CrossEntropyLoss()
    soft_out = nn.Softmax(dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(epochs):
        train_acc = test_acc = 0
        loss_list = []
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = soft_out(model(inputs))
            pred = torch.argmax(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            dist = torch.abs(labels - pred)
            train_acc += len(dist[dist == 0])
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = soft_out(model(inputs))
                pred = torch.argmax(outputs, dim=1)
                dist = torch.abs(labels - pred)
                test_acc += len(dist[dist == 0])
        # print(f"Execution time:{int(hours)} hrs, {int(minutes)} mins, {seconds:.2f} secs")

        # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Train Accuracy: {train_acc/len(filtered_train_dataset)*100:.2f}%, Test Accuracy: {test_acc/len(filtered_test_dataset)*100:.2f}%")
end = time.time()
elapsed_time = end - start
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Execution time:{int(hours)} hrs, {int(minutes)} mins, {seconds:.2f} secs")