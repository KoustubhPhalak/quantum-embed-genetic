'''Use this for analyzing data in plotter.ipynb'''
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd

with open('brute_force_acc.txt', 'r') as f:
    lines = f.readlines()

# Define all permutations
indices = list(range(6))
all_permutations = list(itertools.permutations(indices))

# MNIST data
training_accuracies = list(map(float, lines[26].split(", ")))
inferencing_accuracies = list(map(float, lines[28].split(", ")))
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

# top 3,5,4,2,0,1

# print(sorted_data.loc[sorted_data['Permutations'] == (3,0,2,1,4,5)])
# print(sorted_data.loc[sorted_data['Permutations'] == (0,3,2,1,4,5)])
# print(sorted_data.loc[sorted_data['Permutations'] == (2,1,3,0,4,5)])
# print(sorted_data.loc[sorted_data['Permutations'] == (2,1,0,3,4,5)])
# print(sorted_data.loc[sorted_data['Permutations'] == (2,1,4,5,3,0)])
# print(sorted_data.loc[sorted_data['Permutations'] == (2,1,4,5,0,3)])
# print(sorted_data.loc[sorted_data['Permutations'] == (3,2,1,4,5,0)])



# Get the top 10
top_10_data = sorted_data.head(50)

print(top_10_data)