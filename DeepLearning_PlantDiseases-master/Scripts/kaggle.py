import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchvision import datasets, models
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA

# Settings
data_dir = "D:/Final_Year_Project/Kaggle/houseplant_images"
batch_size = 32
image_size = (64, 64)
num_classes = 2  # healthy and wilted
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stats = []

# Transforms
transform_dl = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_ml = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

# Dataset loader
train_dataset = datasets.ImageFolder(data_dir, transform=transform_dl)
train_loader_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_dataset_ml = datasets.ImageFolder(data_dir, transform=transform_ml)
train_loader_ml = DataLoader(train_dataset_ml, batch_size=batch_size, shuffle=False)

# CNN training and evaluation
def train_evaluate_cnn():
    print("\nTraining CNN (ResNet34)...")
    weights = models.ResNet34_Weights.DEFAULT
    model = models.resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    for epoch in range(1):
        print(f"\nEpoch {epoch+1}/1")
        for i, (images, labels) in enumerate(train_loader_dl):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"  #@ Batch {i}/{len(train_loader_dl)} | Loss: {loss.item():.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in train_loader_dl:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    print("\u2705 CNN F1-Score:", f1)
    stats.append({'name': 'CNN', 'f1_score': f1, 'retrained': True})

# Flatten tensor images to vectors for ML models
def flatten_dataset(loader):
    X, y = [], []
    for imgs, labels in loader:
        flat = imgs.view(imgs.size(0), -1)
        X.append(flat.numpy())
        y.append(labels.numpy())
    return np.concatenate(X), np.concatenate(y)

# SVM classifier
def train_evaluate_svm(X, y):
    print("\nTraining SVM on raw pixels with PCA...")
    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(X)
    svm = SVC(kernel='rbf', gamma='scale')
    svm.fit(X_pca, y)
    preds = svm.predict(X_pca)
    f1 = f1_score(y, preds, average='weighted')
    print("\u2705 SVM F1-Score:", f1)
    stats.append({'name': 'SVM', 'f1_score': f1, 'retrained': False})

# KNN classifier
def train_evaluate_knn(X, y):
    print("\nTraining K-NN on raw pixels with PCA...")
    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(X)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_pca, y)
    preds = knn.predict(X_pca)
    f1 = f1_score(y, preds, average='weighted')
    print("\u2705 K-NN F1-Score:", f1)
    stats.append({'name': 'K-NN', 'f1_score': f1, 'retrained': False})

# Plotting results
def plot_f1_scores():
    names = [s['name'] for s in stats]
    scores = [s['f1_score'] for s in stats]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, scores, color=['blue', 'orange', 'green'])
    plt.ylabel("F1-Score")
    plt.title("Classifier F1-Score Comparison")
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{score:.2f}", ha='center')
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Main
if __name__ == '__main__':
    train_evaluate_cnn()
    X_ml, y_ml = flatten_dataset(train_loader_ml)
    train_evaluate_svm(X_ml, y_ml)
    train_evaluate_knn(X_ml, y_ml)
    plot_f1_scores()
