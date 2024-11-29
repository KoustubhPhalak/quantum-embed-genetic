import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from utils import set_seed
import pennylane as qml
import torch.nn as nn
import time
import itertools

set_seed(123)

# Load the compressed data from .npy files
compressed_imagenet = np.load('reduced_images.npy')
labels = np.load('labels.npy')

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
    torch.manual_seed(123)

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
    soft_out = nn.Softmax(dim=0)
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

# Print results for all index cases
print(f"Training accuracy values: {tr_acc_list}")
print(f"Inferencing accuracy values: {inf_acc_list}")

# Store results in a file
with open("brute_force_acc.txt",'a') as f:
    f.write("Training accuracy values:\n")
    for tr_acc in tr_acc_list:
        f.write(str(tr_acc)+", ")
    f.write('\n')
    f.write("Inferencing accuracy values:\n")
    for inf_acc in inf_acc_list:
        f.write(str(inf_acc)+", ")
    f.write("\n")
    f.write("Index permutations:\n")
    for ind_list in list(itertools.permutations(indices)):
        f.write(str(ind_list)+", ")
    f.close()

