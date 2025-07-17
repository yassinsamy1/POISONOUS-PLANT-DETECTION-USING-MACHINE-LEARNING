import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import models, transforms
from PIL import Image
import os

# Image path
img_path = "D:/Final_Year_Project/PlantVillage/PlantVillage/train/Pepper,_bell___Bacterial_spot/0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG"

# Verify image path
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found at {img_path}")

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image = Image.open(img_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Load model and register hook
try:
    model = models.resnet34(weights="IMAGENET1K_V1")  # Adjust weights argument for compatibility
except AttributeError:
    model = models.resnet34(pretrained=True)  # Fallback for older PyTorch versions
model.eval()

activation = []

def hook_fn(module, input, output):
    activation.append(output)

model.layer1.register_forward_hook(hook_fn)

# Forward pass
with torch.no_grad():
    _ = model(input_tensor)

# Ensure activation tensor has expected dimensions
if len(activation) == 0 or len(activation[0].shape) < 3:
    raise ValueError("Unexpected activation tensor shape. Ensure the model and input are correct.")

act = activation[0][0]  # Take first batch

# Animate
fig, ax = plt.subplots()
frame = ax.imshow(act[0].cpu().numpy(), cmap='plasma')  # Convert to numpy for compatibility
ax.set_title("Animated Filters")

def update(i):
    frame.set_data(act[i].cpu().numpy())  # Convert to numpy for compatibility
    ax.set_title(f"Filter {i}")
    return frame,

ani = animation.FuncAnimation(fig, update, frames=min(32, act.shape[0]), interval=200, blit=False)
plt.show()


#Animated Layers
