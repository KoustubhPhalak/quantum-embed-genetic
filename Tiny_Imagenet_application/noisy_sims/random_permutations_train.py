'''
Code to train 100 random permutations of QNN on the compressed Tiny ImageNet dataset
'''

import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
import torch.nn as nn
import time
import itertools
from utils import *
import os

set_seed(123)

# Load the compressed data from .npy files. Choose either 'reduced_images.npy' or 'reduced_images_hi.npy'
# reduced_images.npy: bell pepper - orange
# reduced_images_hi.npy: sulphur butterfly - alp
compressed_imagenet = np.load('reduced_images_hi.npy')
labels = np.load('labels_hi.npy')

# Convert labels to 0 and 1 by using np.unique to map the unique labels to 0 and 1
unique_labels = np.unique(labels)
label_map = {unique_labels[0]: 0, unique_labels[1]: 1}

# Apply the mapping to the labels
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

# Convert the filtered data and labels to torch tensors
filtered_train_data_tensor = torch.tensor(compressed_train, dtype=torch.float32)
filtered_train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

filtered_test_data_tensor = torch.tensor(compressed_test, dtype=torch.float32)
filtered_test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

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
n_features = 2 * n_qubits
range_param = 1

if not os.path.exists(f"./{n_qubits}q_results"):
    os.makedirs(f"./{n_qubits}q_results")

save_dir = f"./{n_qubits}q_results"

# Create a quantum and classical device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = qml.device("default.qubit", wires=n_qubits)

indices = list(range(n_features))
if n_qubits <= 5:
    all_perm = list(itertools.permutations(indices))

# Select 100 random permutations
random_perms = []
for _ in range(100):
    if n_qubits <= 5:
        random_perm_idx = random.randint(0, len(all_perm)-1)
        random_perms.append(all_perm[random_perm_idx])
    else:
        random_perm = random.sample(indices, n_features)
        random_perms.append(random_perm)

tr_acc_list = []
inf_acc_list = []

for case, ind_list in enumerate(random_perms):
    print("**************************************")
    print(f"Case {case+1}: {ind_list}")
    print("**************************************")

    set_seed(123)

    case_training_acc = []
    case_inferencing_acc = []
    for _ in range(10):
        # Define the quantum circuit
        @qml.qnode(dev, interface='torch')
        def qcircuit(inputs, params):
            custom_rx(inputs[:,ind_list[0]],wires=0)
            custom_ry(inputs[:,ind_list[1]],wires=0)
            custom_rx(inputs[:,ind_list[2]],wires=1)
            custom_ry(inputs[:,ind_list[3]],wires=1)
            custom_rx(inputs[:,ind_list[4]],wires=2)
            custom_ry(inputs[:,ind_list[5]],wires=2)
            if n_qubits in list(range(4, 9)):
                custom_rx(inputs[:,ind_list[6]],wires=3)
                custom_ry(inputs[:,ind_list[7]],wires=3)
            if n_qubits in list(range(5, 9)):
                custom_rx(inputs[:,ind_list[8]],wires=4)
                custom_ry(inputs[:,ind_list[9]],wires=4)
            if n_qubits in list(range(6, 9)):
                custom_rx(inputs[:,ind_list[10]],wires=5)
                custom_ry(inputs[:,ind_list[11]],wires=5)
            if n_qubits in list(range(7, 9)):
                custom_rx(inputs[:,ind_list[12]],wires=6)
                custom_ry(inputs[:,ind_list[13]],wires=6)
            if n_qubits == 8:
                custom_rx(inputs[:,ind_list[14]],wires=7)
                custom_ry(inputs[:,ind_list[15]],wires=7)
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

        # Insert SWAPs
        transpiled_ckt = qml.transforms.transpile(qcircuit, layout_coupling)
        qcircuit = qml.QNode(transpiled_ckt, dev)

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
            if epoch == epochs - 1:
                case_training_acc.append(train_acc/len(filtered_train_dataset)*100)
                case_inferencing_acc.append(test_acc/len(filtered_test_dataset)*100)
    
    # Save final train and inferencing accuracies
    tr_acc_list.append(sum(case_training_acc)/len(case_training_acc))
    inf_acc_list.append(sum(case_inferencing_acc)/len(case_inferencing_acc))
    if not os.path.exists(f"{save_dir}/random_perms_hi.txt"):
        with open(f"{save_dir}/random_perms_hi.txt", 'w+') as f:
            lines = f.readlines()
    else: 
        with open(f"{save_dir}/random_perms_hi.txt", 'r') as f:
            lines = f.readlines()
    if case + 1 == 1: # First permutation
        if (len(lines) < 2) or (lines[0] != "" and lines[1] != ""):
            lines.insert(0, "test")
            lines.insert(0, "")
            lines.insert(0, "")
    if case + 1 < len(random_perms): # First P-1 permutations
        lines[0] = lines[0].strip() + str(tr_acc_list[-1])+',\n'
        lines[1] = lines[1].strip() + str(inf_acc_list[-1])+',\n'
    else: # Final permutation
        lines[0] = lines[0].strip() + str(tr_acc_list[-1])+'\n'
        lines[1] = lines[1].strip() + str(inf_acc_list[-1])+'\n'
    with open(f"{save_dir}/random_perms_hi.txt", 'w') as f: # Write lines
        f.writelines(lines)

# Store randomly selected permutations
with open(f"{save_dir}/random_perms_hi.txt", 'r') as f:
    lines = f.readlines()
    print(len(lines))
    lines[2] = ""
    for i in range(100):
        if i < 99:
            lines[2] = lines[2]+ str(random_perms[i])+","
        else: # Final permutation
            lines[2] = lines[2]+ str(random_perms[i])
    f.close()

with open(f"{save_dir}/random_perms_hi.txt", 'w') as f:
    f.writelines(lines)

# Print results for all index cases
print(f"Training accuracy values: {tr_acc_list}")
print(f"Inferencing accuracy values: {inf_acc_list}")

