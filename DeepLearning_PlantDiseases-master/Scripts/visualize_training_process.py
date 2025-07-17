import matplotlib.pyplot as plt
import numpy as np

# Example dummy data for visualization (replace with real logs if available)
cnn_loss_per_batch = np.random.uniform(0.5, 1.5, 30)  # Simulated loss per batch
cnn_f1_per_epoch = [0.65]  # Simulated F1-score per epoch

svm_training_time = 12.3  # seconds (example)
svm_f1 = 0.72

knn_training_time = 3.8  # seconds (example)
knn_f1 = 0.68

# --- CNN Training Visualization ---
# Schematic: CNN training process (forward, loss, backward, update)
plt.figure(figsize=(8, 2))
plt.axis('off')
plt.text(0.05, 0.5, "Input Image", fontsize=12, va='center')
plt.arrow(0.18, 0.5, 0.1, 0, head_width=0.05, head_length=0.03, fc='k', ec='k')
plt.text(0.32, 0.5, "CNN Layers\n(Conv, Pool, etc.)", fontsize=12, va='center')
plt.arrow(0.48, 0.5, 0.1, 0, head_width=0.05, head_length=0.03, fc='k', ec='k')
plt.text(0.62, 0.5, "Output\n(Prediction)", fontsize=12, va='center')
plt.arrow(0.75, 0.5, 0.1, 0, head_width=0.05, head_length=0.03, fc='k', ec='k')
plt.text(0.87, 0.5, "Loss\nComputation\n& Backprop", fontsize=12, va='center')
plt.title("CNN Training Process Schematic")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(cnn_loss_per_batch, marker='o')
plt.title("CNN Training Loss per Batch")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cnn_f1_per_epoch)+1), cnn_f1_per_epoch, marker='s', color='green')
plt.title("CNN F1-Score per Epoch")
plt.xlabel("Epoch")
plt.ylabel("F1-Score")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.suptitle("CNN Training Process")
plt.show()

# --- SVM Training Visualization ---
# Schematic: SVM training process (feature extraction, margin, support vectors)
plt.figure(figsize=(8, 2))
plt.axis('off')
plt.text(0.05, 0.5, "Input Image\n(flattened)", fontsize=12, va='center')
plt.arrow(0.18, 0.5, 0.1, 0, head_width=0.05, head_length=0.03, fc='k', ec='k')
plt.text(0.32, 0.5, "Feature\nVector", fontsize=12, va='center')
plt.arrow(0.48, 0.5, 0.1, 0, head_width=0.05, head_length=0.03, fc='k', ec='k')
plt.text(0.62, 0.5, "SVM\nClassifier", fontsize=12, va='center')
plt.arrow(0.75, 0.5, 0.1, 0, head_width=0.05, head_length=0.03, fc='k', ec='k')
plt.text(0.87, 0.5, "Finds\nOptimal Margin\n& Support Vectors", fontsize=12, va='center')
plt.title("SVM Training Process Schematic")
plt.tight_layout()
plt.show()

# The following bar chart shows the total training time (in seconds) required for the SVM model to fit the data.
plt.figure(figsize=(6, 4))
plt.bar(['SVM'], [svm_training_time], color='orange')
plt.ylabel("Training Time (s)")
plt.title("SVM Training Time")
plt.tight_layout()
plt.show()

# This bar chart shows the F1-Score achieved by the SVM model on the validation/test set.
plt.figure(figsize=(6, 4))
plt.bar(['SVM'], [svm_f1], color='orange')
plt.ylabel("F1-Score")
plt.title("SVM F1-Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# --- K-NN Training Visualization ---
# Schematic: K-NN training process (feature extraction, storing, neighbor search)
plt.figure(figsize=(8, 2))
plt.axis('off')
plt.text(0.05, 0.5, "Input Image\n(flattened)", fontsize=12, va='center')
plt.arrow(0.18, 0.5, 0.1, 0, head_width=0.05, head_length=0.03, fc='k', ec='k')
plt.text(0.32, 0.5, "Feature\nVector", fontsize=12, va='center')
plt.arrow(0.48, 0.5, 0.1, 0, head_width=0.05, head_length=0.03, fc='k', ec='k')
plt.text(0.62, 0.5, "Store\nAll Vectors", fontsize=12, va='center')
plt.arrow(0.75, 0.5, 0.1, 0, head_width=0.05, head_length=0.03, fc='k', ec='k')
plt.text(0.87, 0.5, "During Prediction:\nFind K Nearest\nNeighbors", fontsize=12, va='center')
plt.title("K-NN Training Process Schematic")
plt.tight_layout()
plt.show()

# The following bar chart shows the total training time (in seconds) required for the K-NN model to fit the data.
plt.figure(figsize=(6, 4))
plt.bar(['K-NN'], [knn_training_time], color='purple')
plt.ylabel("Training Time (s)")
plt.title("K-NN Training Time")
plt.tight_layout()
plt.show()

# This bar chart shows the F1-Score achieved by the K-NN model on the validation/test set.
plt.figure(figsize=(6, 4))
plt.bar(['K-NN'], [knn_f1], color='purple')
plt.ylabel("F1-Score")
plt.title("K-NN F1-Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
