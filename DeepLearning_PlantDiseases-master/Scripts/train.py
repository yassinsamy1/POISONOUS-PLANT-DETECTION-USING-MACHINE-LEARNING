# Full version: Deep Learning (CNN) + Traditional ML (SVM, KNN) comparison
#Average Training Time CNN = 25s-29s

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchvision import datasets, models
from torch.utils.data import DataLoader, Subset
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA

# Settings
data_dir = "D:/Final_Year_Project/PlantVillage/PlantVillage"
batch_size = 32
image_size = (64, 64)
num_classes = 39
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

# Load datasets
train_dl_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_dl)
val_dl_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_dl)
train_ml_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_ml)
val_ml_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_ml)

train_loader_dl = DataLoader(train_dl_dataset, batch_size=batch_size, shuffle=True)
val_loader_dl = DataLoader(val_dl_dataset, batch_size=batch_size, shuffle=False)
train_loader_ml = DataLoader(train_ml_dataset, batch_size=batch_size, shuffle=False)
val_loader_ml = DataLoader(val_ml_dataset, batch_size=batch_size, shuffle=False)

# CNN training
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

    # Save the trained model
    torch.save(model.state_dict(), "cnn_model.pt")
    print("\u2705 Model saved as 'cnn_model.pt'")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader_dl:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    # For CNN, loss values are not tracked here, so set to 0.0
    max_loss = min_loss = avg_loss = 0.0
    training_time = 0.0  # You can add timing if needed
    stats.append({
        'name': 'CNN',
        'f1_score': f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'training_time': training_time,
        'max_loss': max_loss,
        'min_loss': min_loss,
        'avg_loss': avg_loss,
        'retrained': True,
        'y_true': ','.join(map(str, all_labels)),
        'y_pred': ','.join(map(str, all_preds))
    })
    print("\u2705 CNN F1-Score:", f1)
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

# Flatten tensor images to vectors for ML models
def flatten_dataset(loader):
    X, y = [], []
    for imgs, labels in loader:
        flat = imgs.view(imgs.size(0), -1)
        X.append(flat.numpy())
        y.append(labels.numpy())
    return np.concatenate(X), np.concatenate(y)

# SVM classifier
def train_evaluate_svm(X_train, y_train, X_test, y_test):
    import time
    print("\nTraining SVM on raw pixels with PCA...")
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    start_time = time.time()
    svm = SVC(kernel='rbf', gamma='scale', probability=True)
    svm.fit(X_train_pca, y_train)
    training_time = time.time() - start_time

    # Save the trained SVM model and PCA
    import joblib
    joblib.dump({'model': svm, 'pca': pca}, "svm_model.joblib")
    print("\u2705 Model saved as 'svm_model.joblib'")

    preds = svm.predict(X_test_pca)
    f1 = f1_score(y_test, preds, average='weighted')
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    max_loss = min_loss = avg_loss = 0.0
    stats.append({
        'name': 'SVM',
        'f1_score': f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'training_time': training_time,
        'max_loss': max_loss,
        'min_loss': min_loss,
        'avg_loss': avg_loss,
        'retrained': False,
        'y_true': ','.join(map(str, y_test)),
        'y_pred': ','.join(map(str, preds))
    })
    print(f"\u2705 SVM F1-Score: {f1}")

# KNN classifier
def train_evaluate_knn(X_train, y_train, X_test, y_test):
    import time
    print("\nTraining K-NN on raw pixels with PCA...")
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)
    training_time = time.time() - start_time

    # Save the trained KNN model and PCA
    import joblib
    joblib.dump({'model': knn, 'pca': pca}, "knn_model.joblib")
    print("\u2705 Model saved as 'knn_model.joblib'")

    preds = knn.predict(X_test_pca)
    f1 = f1_score(y_test, preds, average='weighted')
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    max_loss = min_loss = avg_loss = 0.0
    stats.append({
        'name': 'K-NN',
        'f1_score': f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'training_time': training_time,
        'max_loss': max_loss,
        'min_loss': min_loss,
        'avg_loss': avg_loss,
        'retrained': False,
        'y_true': ','.join(map(str, y_test)),
        'y_pred': ','.join(map(str, preds))
    })
    print(f"\u2705 K-NN F1-Score: {f1}")

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

# === Main pipeline ===
if __name__ == '__main__':
    train_evaluate_cnn()

    print("\nPreparing data for SVM and K-NN classifiers...")
    X_train_ml, y_train_ml = flatten_dataset(train_loader_ml)
    X_val_ml, y_val_ml = flatten_dataset(val_loader_ml)

    train_evaluate_svm(X_train_ml, y_train_ml, X_val_ml, y_val_ml)
    train_evaluate_knn(X_train_ml, y_train_ml, X_val_ml, y_val_ml)

    plot_f1_scores()

    # Save stats to CSV
    import pandas as pd
    stats_path = "d:\\Final_Year_Project\\DeepLearning_PlantDiseases-master\\DeepLearning_PlantDiseases-master\\Scripts\\stats.csv"
    df_stats = pd.DataFrame(stats)
    # Ensure columns order
    columns = [
        'name', 'f1_score', 'accuracy', 'precision', 'recall', 'training_time',
        'max_loss', 'min_loss', 'avg_loss', 'retrained', 'y_true', 'y_pred'
    ]
    df_stats = df_stats.reindex(columns=columns)
    df_stats.to_csv(stats_path, index=False)
    print(f"\n\u2705 Stats saved to {stats_path}")

    # Visualize confusion matrices and metrics
    from visual_comparison import plot_confusion_matrices, plot_model_metrics
    # Prepare for plotting
    y_trues = [list(map(int, s['y_true'].split(','))) for s in stats]
    y_preds = [list(map(int, s['y_pred'].split(','))) for s in stats]
    model_names = [s['name'] for s in stats]
    labels_set = set()
    for yt, yp in zip(y_trues, y_preds):
        labels_set.update(yt)
        labels_set.update(yp)
    labels = sorted(labels_set)
    plot_confusion_matrices(y_trues, y_preds, model_names, labels=labels)
    metrics = {s['name']: {
        'accuracy': s['accuracy'],
        'precision': s['precision'],
        'recall': s['recall'],
        'f1': s['f1_score']
    } for s in stats}
    plot_model_metrics(metrics)