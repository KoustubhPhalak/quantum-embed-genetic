'''Use this for analyzing data in plotter.ipynb'''
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd

cls1 = int(input("Enter first class: "))
cls2 = int(input("Enter second class: "))
n_qubits = int(input("Enter number of qubits: "))

with open(f'{n_qubits}q_results/brute_force_{cls1}_{cls2}.txt', 'r') as f:
    lines = f.readlines()

# Define permutations
indices = list(range(6))
all_permutations = list(itertools.permutations(indices))

## USE BELOW LINES FOR RANDOM PERMUTATIONS
# cleaned_input = lines[2].strip().replace('(', '').replace(')', '').replace(' ', '')
# split_values = cleaned_input.split(',')
# all_permutations = [tuple(map(int, split_values[i:i + 2*n_qubits])) for i in range(0, len(split_values), 2*n_qubits)]

# MNIST data
training_accuracies = list(map(float, lines[0].strip().split(",")))
inferencing_accuracies = list(map(float, lines[1].strip().split(",")))
combined_scores = [(t+i)/2 for t,i in zip(training_accuracies, inferencing_accuracies)]

# Combine the data into a DataFrame for easy sorting
data = pd.DataFrame({
    'Permutations': all_permutations,
    'Training Accuracies': training_accuracies,
    'Inferencing Accuracies': inferencing_accuracies,
    'Combined Score': combined_scores
})

# Sort by descending order of inferencing accuracies
sorted_data = data.sort_values(by='Combined Score', ascending=False)
sorted_data = sorted_data.reset_index(drop=True)

# Check random cases (can modify as needed)
# print(sorted_data.loc[sorted_data['Permutations'] == (3,0,2,1,4,5)])
# print(sorted_data.loc[sorted_data['Permutations'] == (0,3,2,1,4,5)])
# print(sorted_data.loc[sorted_data['Permutations'] == (2,1,3,0,4,5)])
# print(sorted_data.loc[sorted_data['Permutations'] == (2,1,0,3,4,5)])
# print(sorted_data.loc[sorted_data['Permutations'] == (2,1,4,5,3,0)])
# print(sorted_data.loc[sorted_data['Permutations'] == (2,1,4,5,0,3)])
print(sorted_data.loc[sorted_data['Permutations'] == (3,5,4,2,0,1)])



# Get the top 10
print(sorted_data.head(10))
print(sorted_data.tail(10))