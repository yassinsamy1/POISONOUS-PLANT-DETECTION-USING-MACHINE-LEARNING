# CODE BOOKLET

## POISONOUS PLANT DETECTION USING MACHINE LEARNING

**ACADEMIC SESSION:** 2024/2025  
**STUDENT:** Yassin Mahmoud Samy (A21EC0282)  
**PROGRAM:** Bachelor of Computer Science (Software Engineering)  
**SUPERVISOR:** DR. Hishammuddin Asmuni  
**CO-SUPERVISOR:**  

---

### SUMMARY
This research presents all the work and implementation behind the final year project POISONOUS PLANT DETECTION USING MACHINE LEARNING, which includes the machine learning models, preprocessing pipelines, evaluation metrics, and web-based user interface. The system addresses the critical challenge of accurately distinguishing poisonous from non-poisonous plants, a task that is traditionally manual and time-consuming. By leveraging deep learning (CNN) and traditional models (SVM, K-NN), the project offers an automated, reliable solution that aids farmers, researchers, and the general public in making safer, data-driven decisions about plant identification.

---

## Table of Contents

1. Project Hardware and Software List
2. Modules List and Explanation
3. Code for Plant Disease Detection UI
4. Code for Model Training & Evaluation
5. Code for Data Preprocessing & Visualization
6. Code for Feature Extraction & Visualization
7. Code for Model Performance Visualization
8. User Manual

---

## 1. Project Hardware and Software List

| HARDWARE | VERSION | EXPLANATION OF USAGE |
|----------|---------|----------------------|
| PC/Laptop | Any modern x64 | Used for model training, evaluation, and running the web interface |
| (Optional) GPU | CUDA-enabled | Accelerates deep learning model training |

| SOFTWARE | VERSION | EXPLANATION OF USAGE |
|----------|---------|----------------------|
| Python | 3.8+ | Main programming language |
| PyTorch | 2.x | Deep learning framework for CNN models |
| scikit-learn | 1.x | Traditional ML models (SVM, K-NN) |
| Gradio | 3.x | Web-based user interface for plant disease detection |
| matplotlib | 3.x | Visualization of results and training |
| numpy, pandas | 1.x | Data processing and analysis |

---

## 2. Modules List and Explanation

| MODULE | FUNCTION | USERS |
|--------|----------|-------|
| Plant Disease Detection UI (`plant_ui.py`) | Web interface for uploading plant images and getting disease predictions using a trained CNN model | End users (farmers, researchers, public) |
| Model Training & Evaluation (`train.py`, `kaggle.py`) | Train and evaluate CNN, SVM, and K-NN models on plant datasets | Researcher/Developer |
| Data Preprocessing & Visualization (`visualize_preprocessing.py`) | Preprocess images for CNN/ML and visualize the transformations | Researcher/Developer |
| Feature Extraction & Visualization (`feature_extraction_visualization.py`, `Flavia.py`) | Extract and visualize features from images for ML models and unsupervised tasks | Researcher/Developer |
| Model Performance Visualization (`plot.py`, `visual_comparison.py`, `visualize_training_process.py`) | Visualize model metrics, confusion matrices, and training process | Researcher/Developer |

---

## 3. Code for Plant Disease Detection UI

**File:** `Scripts/plant_ui.py`

This module provides a web-based interface using Gradio for users to upload plant leaf images and receive disease predictions from a trained CNN model (ResNet34). The UI is styled for clarity and ease of use.

```python
# Load the trained CNN model
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction function for Gradio
def predict_plant(image):
    img = Image.fromarray(image).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_index].item()
    predicted_label = class_names[predicted_index]
    return {predicted_label: confidence}

# Gradio UI setup (see full code in plant_ui.py)
```

---

## 4. Code for Model Training & Evaluation

**Files:** `Scripts/train.py`, `Scripts/kaggle.py`

These modules handle the training and evaluation of deep learning (CNN) and traditional ML (SVM, K-NN) models. They include data loading, preprocessing, model definition, training loops, evaluation, and saving of results.

**Key Functions:**
- `train_evaluate_cnn()`: Trains and evaluates a ResNet34 CNN model.
- `train_evaluate_svm()`: Trains and evaluates an SVM classifier with PCA.
- `train_evaluate_knn()`: Trains and evaluates a K-NN classifier with PCA.
- `flatten_dataset()`: Flattens image tensors for ML models.

```python
# Example: Training CNN
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
# ... training loop ...
torch.save(model.state_dict(), "cnn_model.pt")

# Example: Training SVM
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
svm = SVC(kernel='rbf', gamma='scale', probability=True)
svm.fit(X_train_pca, y_train)
joblib.dump({'model': svm, 'pca': pca}, "svm_model.joblib")

# Example: Training K-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)
joblib.dump({'model': knn, 'pca': pca}, "knn_model.joblib")
```

---

## 5. Code for Data Preprocessing & Visualization

**File:** `Scripts/visualize_preprocessing.py`

This module demonstrates how images are preprocessed for both CNN and traditional ML models, and visualizes the transformations.

```python
# Preprocessing for CNN
transform_cnn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Preprocessing for SVM/K-NN
transform_ml = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Visualization code plots original, CNN-preprocessed, and SVM/K-NN preprocessed images
```

---

## 6. Code for Feature Extraction & Visualization

**Files:** `Scripts/feature_extraction_visualization.py`, `Scripts/Flavia.py`

These modules show how features are extracted from images for use in ML models, and how to visualize them. The Flavia module also demonstrates unsupervised feature extraction for clustering or visualization.

```python
# SVM/K-NN feature extraction
img_tensor_ml = transform_ml(img)
flat_features = img_tensor_ml.view(-1).numpy()

# CNN feature extraction (first conv layer)
img_tensor_cnn = transform_cnn(img).unsqueeze(0)
model = models.resnet34(weights=None)
with torch.no_grad():
    features = model.conv1(img_tensor_cnn)
```

---

## 7. Code for Model Performance Visualization

**Files:** `Scripts/plot.py`, `Scripts/visual_comparison.py`, `Scripts/visualize_training_process.py`

These modules visualize model performance using bar charts, confusion matrices, and training process diagrams.

```python
# Plotting metrics (plot.py)
fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
# ...

# Confusion matrix (visual_comparison.py)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_trues[i], y_preds[i], labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=axes[i], cmap='Blues', colorbar=False)

# Training process visualization (visualize_training_process.py)
plt.plot(cnn_loss_per_batch, marker='o')
plt.title("CNN Training Loss per Batch")
```

---

## 8. User Manual

**How to Use the System:**
1. Train the models using `train.py` (see README for dataset requirements).
2. Run `plant_ui.py` to launch the web interface and upload plant images for prediction.
3. Use the visualization scripts to analyze model performance and understand the results.

**Note:** For more details, see the README.md and comments in each script.

---

**SCHOOL OF COMPUTING**

2024

CB_V1_2024 