
# Deep Learning & Traditional ML for Plant Disease Classification

This project compares deep learning (CNN) and traditional machine learning (SVM, K-NN) approaches for classifying plant diseases using leaf images. It includes training, evaluation, visualization, and an interactive UI for predictions.

## Project Structure
- **Scripts/train.py**: Main script for training and evaluating CNN, SVM, and K-NN models on the PlantVillage dataset. Includes functions for training, evaluation, plotting metrics, and saving results.
- **Scripts/plant_ui.py**: Interactive Gradio web app for plant disease prediction. Users can upload leaf images and select a model for prediction.
- **Scripts/kaggle.py**: Training and evaluation on a Kaggle houseplant dataset (healthy vs. wilted). Demonstrates binary classification and model comparison.
- **Scripts/CNN_Healthy_Classification_Visualization.py**: Visualizes the step-by-step process of CNN classification using diagrams.
- **Scripts/feature_extraction_visualization.py**: Shows how features are extracted for SVM/K-NN (flattened vectors) and CNN (feature maps).
- **Scripts/Visualize2.py**: Animates CNN feature maps for a sample image, helping to understand filter activations.
- **Scripts/Visualize.py**: Compares CNN activations for healthy and diseased leaves, visualizing differences in learned features.
- **Scripts/visualize_training_process.py**: Plots training curves (loss, F1-score) and schematic diagrams for all models, illustrating the training workflow.
- **Scripts/visualize_preprocessing.py**: Visualizes image preprocessing steps for each model, showing how input images are transformed.

## Dataset
- **PlantVillage**: Contains images of plant leaves categorized by disease type. Split into `train` and `val` folders for training and validation.
- **Kaggle houseplant dataset**: Used for binary classification (healthy vs. wilted).

## How to Run
1. Place the PlantVillage dataset in the correct directory (`PlantVillage/PlantVillage/train` and `val`).
2. Run `train.py` to train and evaluate all models. Results and trained models will be saved in the project folder.
3. Use `plant_ui.py` to launch the interactive prediction interface. Upload a leaf image and select a model to get predictions.
4. Explore visualization scripts to gain insights into model behavior, feature extraction, and training dynamics.

## Key Features
- End-to-end training and evaluation of CNN, SVM, and K-NN models.
- Visual comparison of model performance (F1-score, accuracy, precision, recall) using plots and confusion matrices.
- Feature extraction and activation visualizations to understand how models process images.
- Interactive UI for real-time plant disease predictions.
- Support for both multiclass (PlantVillage) and binary (Kaggle) classification tasks.

## Requirements
- Python 3.8+
- PyTorch, torchvision, scikit-learn, matplotlib, PIL, gradio
- Download PlantVillage dataset and place in the specified directory structure.

## Results & Analysis
- The project provides a detailed comparison of deep learning and traditional ML for plant disease classification.
- CNN (ResNet34) typically achieves higher accuracy and F1-score compared to SVM and K-NN, especially on complex multiclass tasks.
- Visualizations help explain how CNNs learn features and how traditional ML relies on raw pixel data.
- All results, metrics, and visualizations are saved for reporting and presentation.

## Usage Scenarios
- Academic research and benchmarking of ML models for plant disease detection.
- Educational demonstrations of deep learning vs. traditional ML.
- Practical tool for farmers or researchers to identify plant diseases from leaf images.

---
For more details, see the code comments and individual script files.
