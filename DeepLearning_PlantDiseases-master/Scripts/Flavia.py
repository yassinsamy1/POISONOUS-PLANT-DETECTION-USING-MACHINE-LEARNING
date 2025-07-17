# Flavia Dataset Instructions:
# Download the Flavia leaf dataset from:
# http://flavia.sourceforge.net/
# Unzip and organize the dataset into train/ and test/ folders, each with subfolders for each class (leaf species).
# Example structure:
#   Flavia/
#     train/
#       class1/
#       class2/
#       ...
#     test/
#       class1/
#       class2/
#       ...

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. Data transforms and loading
batch_size = 32
num_workers = 2

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # ...add normalization if needed...
])

# To visualize a few predictions (like MATLAB's comparison):
def show_predictions(model, loader, class_names, device, num_images=5):
    model.eval()
    images_shown = 0
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break
                img = images[i].cpu().permute(1, 2, 0).numpy()
                axes[images_shown].imshow(img)
                axes[images_shown].set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
                axes[images_shown].axis('off')
                images_shown += 1
            if images_shown >= num_images:
                break
    plt.show()

# Usage instructions:
# If your Leaves folder contains only images (no class subfolders), you can:
# - Use it for unsupervised tasks (e.g., clustering, autoencoding, feature extraction)
# - Use it as a single-class dataset for testing model pipelines (no real classification)
# - For supervised learning, you must organize images into subfolders by class

# Example: Use Leaves as a single-class dataset for feature extraction/testing
# If you do not have class labels for each image, you cannot perform supervised classification.
# You can still use the code for:
# - Feature extraction (for clustering or visualization)
# - Unsupervised learning (e.g., clustering, autoencoders)
# - Testing data loading and augmentation

# The SingleClassDataset and feature extraction example below will work without class labels.

class SingleClassDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.labels = [0] * len(self.images)  # All images have label 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def main():
    import sys

    # Set your Leaves folder path here
    data_dir = r"D:\Final_Year_Project\Leaves"

    if not os.path.isdir(data_dir):
        print("Invalid directory. Exiting.")
        sys.exit(1)

    # Use SingleClassDataset for a folder with only images and no subfolders
    dataset = SingleClassDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_classes = 1  # Only one class

    # Example: Feature extraction using a pretrained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer
    model.eval()
    model.to('cpu')

    features = []
    with torch.no_grad():
        for images, _ in loader:
            feats = model(images)
            feats = feats.view(feats.size(0), -1)
            features.append(feats.numpy())
    features = np.concatenate(features)
    print(f"Extracted features shape: {features.shape}")

    # You can now use 'features' for clustering, visualization, or as input to other ML models
    # Supervised classification is NOT possible without class labels.

if __name__ == "__main__":
    main()