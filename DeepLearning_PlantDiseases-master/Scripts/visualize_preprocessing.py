import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch

# Set a sample image path (update this path to an actual image in your dataset)
sample_img_path = "D:/Final_Year_Project/PlantVillage/PlantVillage/train/Tomato___healthy/0cfee8b1-5de4-4118-9e25-f9c37bb6f17e___GH_HL Leaf 252.JPG"


# Preprocessing transform (resize and normalize) for CNN
transform_cnn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Preprocessing transform for SVM/K-NN (resize only, no normalization)
transform_ml = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load original image
img = Image.open(sample_img_path).convert('RGB')

# Preprocess for CNN
img_tensor_cnn = transform_cnn(img)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_np_cnn = img_tensor_cnn.numpy().transpose(1, 2, 0)
img_np_cnn = std * img_np_cnn + mean
img_np_cnn = np.clip(img_np_cnn, 0, 1)

# Preprocess for SVM/K-NN
img_tensor_ml = transform_ml(img)
img_np_ml = img_tensor_ml.numpy().transpose(1, 2, 0)
img_np_ml = np.clip(img_np_ml, 0, 1)

# Plot original, CNN-preprocessed, and SVM/K-NN preprocessed images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title(f"Original\nSize: {img.size}")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_np_cnn)
#plt.title(f"CNN Preprocessed\nSize: {img_tensor_cnn.shape[1:]} (224x224)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_np_ml)
#plt.title(f"SVM/K-NN Preprocessed\nSize: {img_tensor_ml.shape[1:]} (64x64)")
plt.axis('off')

plt.suptitle("Image Before and After Preprocessing (CNN & SVM/K-NN)")
plt.tight_layout()
plt.show()

# --- CNN feature extraction visualization () ---
# import torchvision.models as models
# transform_cnn = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# img_tensor_cnn = transform_cnn(img).unsqueeze(0)
# model = models.resnet34(weights=None)
# with torch.no_grad():
#     features = model.conv1(img_tensor_cnn)
# feature_map = features[0, :6].cpu().numpy()  # Show first 6 feature maps
# plt.figure(figsize=(12, 3))
# for i in range(6):
#     plt.subplot(1, 6, i+1)
#     plt.imshow(feature_map[i], cmap='gray')
#     plt.axis('off')
# plt.suptitle("CNN First Conv Layer Feature Maps (First 6 Channels)")
# plt.show()
