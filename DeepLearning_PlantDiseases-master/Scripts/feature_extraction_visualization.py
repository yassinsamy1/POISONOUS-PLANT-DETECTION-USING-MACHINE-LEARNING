import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.models as models

# Set a sample image path (update this path to an actual image in your dataset)
sample_img_path = "D:/Final_Year_Project/PlantVillage/PlantVillage/train/Potato___Late_blight/1ab62af3-c0a5-4fab-bb62-e06d6f7ddb59___RS_LB 2908.JPG"

# --- SVM Feature Extraction Visualization ---
transform_ml = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
img = Image.open(sample_img_path).convert('RGB')
img_tensor_ml = transform_ml(img)
flat_features = img_tensor_ml.view(-1).numpy()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_tensor_ml.numpy().transpose(1, 2, 0))
plt.title(f"SVM Preprocessed\nSize: {img_tensor_ml.shape[1:]} (64x64)")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.plot(flat_features[:500])
plt.title("SVM Flattened Feature Vector (First 500 values)")
plt.xlabel("Feature Index")
plt.ylabel("Pixel Value")
plt.tight_layout()
plt.suptitle("SVM Feature Extraction Visualization")
plt.show()

# --- K-NN Feature Extraction Visualization ---
# For K-NN, the preprocessing is the same as SVM, but we show a separate diagram for clarity
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_tensor_ml.numpy().transpose(1, 2, 0))
plt.title(f"K-NN Preprocessed\nSize: {img_tensor_ml.shape[1:]} (64x64)")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.plot(flat_features[:500])
plt.title("K-NN Flattened Feature Vector (First 500 values)")
plt.xlabel("Feature Index")
plt.ylabel("Pixel Value")
plt.tight_layout()
plt.suptitle("K-NN Feature Extraction Visualization")
plt.show()

# --- CNN Feature Extraction (First Conv Layer Feature Maps) ---
transform_cnn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img_tensor_cnn = transform_cnn(img).unsqueeze(0)
model = models.resnet34(weights=None)
with torch.no_grad():
    features = model.conv1(img_tensor_cnn)
feature_map = features[0, :6].cpu().numpy()  # Show first 6 feature maps
plt.figure(figsize=(12, 3))
for i in range(6):
    plt.subplot(1, 6, i+1)
    plt.imshow(feature_map[i], cmap='gray')
    plt.axis('off')
plt.suptitle("CNN First Conv Layer Feature Maps (First 6 Channels)")
plt.show()
