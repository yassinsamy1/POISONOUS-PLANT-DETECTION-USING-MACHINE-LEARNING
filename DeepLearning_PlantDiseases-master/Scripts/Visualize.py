import torch
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

def load_image_tensor(path):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Replace these paths
diseased_img_path = "D:/Final_Year_Project/PlantVillage/PlantVillage/train/Tomato___Late_blight/0e669afb-1315-4903-b7ca-1a0ffa95d454___RS_Late.B 5626.JPG"
healthy_img_path = "D:/Final_Year_Project/PlantVillage/PlantVillage/train/Tomato___healthy/0cfee8b1-5de4-4118-9e25-f9c37bb6f17e___GH_HL Leaf 252.JPG"

diseased_image = Image.open(diseased_img_path).convert('RGB')
healthy_image = Image.open(healthy_img_path).convert('RGB')


diseased = load_image_tensor(diseased_img_path)
healthy = load_image_tensor(healthy_img_path)

# Load model
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
model.eval()

layer = model.layer1
diseased_act = []
healthy_act = []

# Hook functions
def hook_diseased(m, i, o):
    diseased_act.append(o)

def hook_healthy(m, i, o):
    healthy_act.append(o)

# Register and run for diseased image
handle1 = layer.register_forward_hook(hook_diseased)
with torch.no_grad():
    _ = model(diseased)
handle1.remove()

# Register and run for healthy image
handle2 = layer.register_forward_hook(hook_healthy)
with torch.no_grad():
    _ = model(healthy)
handle2.remove()


#diseased_act = torch.tensor(diseased_act[0])
diseased_act = diseased_act[0].clone().detach()
healthy_act = torch.tensor(healthy_act[0])

# Visualize side-by-side
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
# Check how many channels are available
num_filters = min(diseased_act.shape[1], healthy_act.shape[1], 5)  # max 5 filters or fewer if not available

fig, axs = plt.subplots(2, num_filters, figsize=(num_filters * 4, 6))
axs[0, 0].imshow(diseased_image)
axs[0, 0].set_title("Diseased Leaf")
axs[0, 0].axis('off')

axs[1, 0].imshow(healthy_image)
axs[1, 0].set_title("Healthy Leaf")
axs[1, 0].axis('off')

for i in range(1, num_filters):
    axs[0, i].imshow(diseased_act[0][i].cpu(), cmap='inferno')
    axs[0, i].set_title(f"Diseased - Filter {i}")
    axs[0, i].axis('off')

    axs[1, i].imshow(healthy_act[0][i].cpu(), cmap='inferno')
    axs[1, i].set_title(f"Healthy - Filter {i}")
    axs[1, i].axis('off')

plt.tight_layout()
plt.show()