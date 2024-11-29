import os
import umap
from torchvision import datasets, transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from itertools import combinations
import warnings
from sklearn.metrics import silhouette_score

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define the output directory for saving the plots
output_dir = "umap_plots"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Define transformations for Tiny ImageNet dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Tiny ImageNet images are 64x64 pixels
    transforms.ToTensor(),
])

# Step 2: Specify the path to the Tiny ImageNet training dataset
data_path = 'tiny-imagenet-200/train'  # Replace with your actual path

# Step 3: Load the dataset using ImageFolder
imagenet_data = datasets.ImageFolder(data_path, transform=transform)

# Step 4: Get all class combinations
classes = imagenet_data.classes  # List of all classes
class_pairs = list(combinations(classes, 2))  # All possible class pairs

silhouette_scores = []

# Step 5: Loop through each pair of classes and perform UMAP reduction
for pair in class_pairs:
    first_two_classes = pair
    print(f"Processing classes: {first_two_classes}")
    
    # Step 6: Get the class indices for the selected class pair
    class_indices = [imagenet_data.class_to_idx[cls] for cls in first_two_classes]
    
    # Step 7: Filter the dataset to include only the selected class pair
    filtered_indices = [
        i for i, (_, label) in enumerate(imagenet_data.samples)
        if label in class_indices
    ]
    
    # Create a subset of the dataset with only the filtered samples
    subset = torch.utils.data.Subset(imagenet_data, filtered_indices)
    
    # Step 8: Create a DataLoader for the subset
    dataloader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False)
    
    # Step 9: Collect images and labels from the subset
    images = []
    labels = []
    for batch in dataloader:
        imgs, lbls = batch
        imgs = imgs.view(imgs.size(0), -1)  # Flatten images to 1D vectors
        images.append(imgs.numpy())
        labels.append(lbls.numpy())
    
    # Concatenate all batches
    images = np.vstack(images)
    labels = np.hstack(labels)
    
    # Step 10: Apply UMAP to reduce dimensions to 6
    umap_model = umap.UMAP(n_components=6, n_neighbors=15, min_dist=0.1, random_state=123)
    reduced_data = umap_model.fit_transform(images)
    ss = silhouette_score(reduced_data, labels)

    silhouette_scores.append(ss)
    print(f"Separation score: {ss}")
    with open('class_selection_log.txt', 'a') as f:
        f.write(f"Classes:{first_two_classes}, Separation Score: {ss}")
        f.write("\n")
        f.close()
