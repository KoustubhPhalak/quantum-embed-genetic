'''
Code to run genetic algorithm for feature selection on the MNIST dataset.
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

# Filter the training data for cls1 and cls2
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
n_features = n_qubits * 2

# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = qml.device("default.qubit", wires=n_qubits)

population_size = 20  # Size of the population
num_generations = 5  # Number of generations to run
mutation_rate = 0.001  # Probability of mutation
crossover_rate = 0.8  # Probability of crossover
top_retention_rate = 0.1

# For 3 qubit cases only with sweep data available
if n_qubits == 3:
    indices = list(range(n_features))
    all_perm = list(itertools.permutations(indices))
    all_perm = [list(i) for i in all_perm]
    with open(f"3q_results/brute_force_{cls1}_{cls2}.txt", 'r') as f:
        lines = f.readlines()

    tr_acc_list = list(map(float, lines[0].strip().split(",")))
    te_acc_list = list(map(float, lines[1].strip().split(",")))

def fitness_function_3q(individual):
    perm_idx = all_perm.index(individual.tolist())
    train_acc = tr_acc_list[perm_idx]
    test_acc = te_acc_list[perm_idx]
    return (train_acc + test_acc)/2


def fitness_function(individual):
    # Assuming `individual` is a tuple representing a permutation of indices
    set_seed(123)

    ind_list = list(individual)
    
    @qml.qnode(dev, interface='torch')
    def qcircuit(inputs, params):
        qml.RX(inputs[:,ind_list[0]], wires=0)
        qml.RY(inputs[:,ind_list[1]], wires=0)
        qml.RX(inputs[:,ind_list[2]], wires=1)
        qml.RY(inputs[:,ind_list[3]], wires=1)
        qml.RX(inputs[:,ind_list[4]], wires=2)
        qml.RY(inputs[:,ind_list[5]], wires=2)
        if n_qubits == 4 or n_qubits == 5:
            qml.RX(inputs[:,ind_list[6]],wires=3)
            qml.RY(inputs[:,ind_list[7]],wires=3)
        if n_qubits == 5:
            qml.RX(inputs[:,ind_list[8]],wires=4)
            qml.RY(inputs[:,ind_list[9]],wires=4)
        qml.templates.StronglyEntanglingLayers(params, wires=range(n_qubits), ranges=[1,1,1])
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_classes)]
    
    weight_shapes = {"params": (n_layers, n_qubits, 3)}

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qml.qnn.TorchLayer(qcircuit, weight_shapes, init_method=torch.nn.init.normal_).cuda()

        def forward(self, x):
            out = self.qlayer(x)
            return out
    
    model = Model().to(device)

    # Training loop (similar to your brute force approach)
    epochs = 5
    loss_fn = nn.CrossEntropyLoss()
    soft_out = nn.Softmax(dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-3)

    train_acc_list = []
    test_correct = 0
    start = time.time()
    for epoch in range(epochs):
        train_loss = 0.0
        train_correct = 0
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = soft_out(model(inputs))
            pred = torch.argmax(outputs, dim=1)
            loss = loss_fn(outputs, labels.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (pred == labels).sum().item()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = soft_out(model(inputs))
        pred = torch.argmax(outputs, dim=1)
        test_correct += (pred == labels).sum().item()

    train_acc = train_correct / len(filtered_train_data)
    test_acc = test_correct / len(filtered_test_data)

    end = time.time()
    elapsed_time = end - start
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Execution time:{int(hours)} hrs, {int(minutes)} mins, {seconds:.2f} secs")
    print(round(seconds*100/60,1), " min")
    exit()
    # Return the average training accuracy as fitness
    return (train_acc + test_acc) / 2

def genetic_algorithm(population_size, num_generations):
    population = [np.random.permutation(range(n_features)) for _ in range(population_size)]
    
    for generation in range(num_generations):
        if n_qubits == 3:
            fitness_scores = [fitness_function_3q(individual) for individual in population]
        else:
            fitness_scores = [fitness_function(individual) for individual in population]
        sorted_indices = np.argsort(fitness_scores)[::-1]

        print(fitness_scores)
        
        # Select parents based on fitness scores (e.g., tournament selection)
        parents = selection(population, fitness_scores)
        
        offspring = []
        for i in range(int(top_retention_rate*population_size)):
            offspring.append(population[sorted_indices[i]])
        while len(offspring) < population_size:
            parent1_index, parent2_index = np.random.choice(len(parents), 2, replace=False)
            parent1, parent2 = parents[parent1_index], parents[parent2_index]
            
            if np.random.rand() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1
            
            if np.random.rand() < mutation_rate:
                child = mutate(child)
            
            offspring.append(child)
        
        population = np.array(offspring)
        print(population)
    
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual

def selection(population, fitness_scores):
    # Tournament selection
    selected_parents = []
    for _ in range(len(population)):
        idx1, idx2 = np.random.choice(range(len(population)), 2, replace=False)
        if fitness_scores[idx1] > fitness_scores[idx2]:
            selected_parents.append(population[idx1])
        else:
            selected_parents.append(population[idx2])
    return selected_parents

def crossover(parent1, parent2):
    # Single-point crossover with unique features only
    point = np.random.randint(0, len(parent1))
    
    # Create a copy of parent1 to avoid modifying the original
    child = parent1[:point]    
    
    # Add non-duplicate features from parent2
    for feature in parent2:
        if feature not in child:
            child = np.concatenate([child, [feature]])
    return np.array(child[:n_features])

def mutate(individual):
    # Swap mutation with unique features only
    idx1, idx2 = np.random.choice(range(len(individual)), 2, replace=False)
    
    # Ensure the swap doesn't introduce duplicates
    if idx1 != idx2:
        temp = individual[idx1]
        individual[idx1] = individual[idx2]
        individual[idx2] = temp
    
    return individual

best_config = genetic_algorithm(population_size, num_generations)
print("Best configuration found:", best_config)

# Evaluate the fitness of the best configuration (optional)
if n_qubits == 3:
    best_fitness = fitness_function_3q(best_config)
else:
    best_fitness = fitness_function(best_config)
print(f"Fitness of best configuration: {best_fitness}")
