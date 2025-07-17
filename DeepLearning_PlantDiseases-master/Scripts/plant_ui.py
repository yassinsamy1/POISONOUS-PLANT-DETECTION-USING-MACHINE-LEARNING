import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import gradio as gr
import os

# === 1. Load model (ResNet34 with custom classifier) ===
num_classes = 39  # adjust if you have different classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Update the model file path to the correct file
model_path = "cnn_model.pt"  # Updated to use the saved model
model.load_state_dict(torch.load(model_path, map_location=device))

model = model.to(device)
model.eval()

# === 2. Define transformation ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 3. Define your label mapping ===
# Dynamically load class names from the dataset
data_dir = "D:/Final_Year_Project/PlantVillage/PlantVillage"
dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'))
class_names = [class_name for class_name, _ in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])]

# === 4. Gradio prediction function ===
def predict_plant(image):
    img = Image.fromarray(image).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_index].item()

    predicted_label = class_names[predicted_index]
    return {predicted_label: confidence}  # For gr.Label

# === 5. Gradio interface ===

custom_css = """
.gradio-container {
    background: linear-gradient(120deg, #f6f9fc 0%, #e9f2ff 100%);
    font-family: 'Segoe UI', Arial, sans-serif;
}

#main-title h1 {
    color: #1a365d;
    font-size: 2.5rem;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

#main-description {
    text-align: center;
    color: #4a5568;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.upload-box {
    border: 2px dashed #cbd5e0;
    border-radius: 12px;
    padding: 20px;
    transition: all 0.3s ease;
}

.upload-box:hover {
    border-color: #4299e1;
    background: rgba(66, 153, 225, 0.05);
}

#detect-btn {
    background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
    border: none;
    border-radius: 8px;
    color: white;
    padding: 12px 24px;
    font-weight: 600;
    transition: transform 0.2s ease;
}

#detect-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(66, 153, 225, 0.3);
}

.output-container {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

footer {display: none;}
"""

with gr.Blocks(css=custom_css, title="Plant Disease Detection") as demo:
    gr.Markdown(
        """
        # 🌿 Plant Disease Detection
        """,
        elem_id="main-title"
    )
    
    gr.Markdown(
        """
        Upload a clear image of a plant leaf and our AI model will analyze it for diseases.
        Get instant results with confidence scores!
        """,
        elem_id="main-description"
    )
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Plant Image",
                type="numpy",
                elem_id="image-upload",
                container=True,
                elem_classes="upload-box"
            )
            submit_btn = gr.Button(
                "🔍 Analyze Plant",
                elem_id="detect-btn",
                size="lg"
            )
        
        with gr.Column(scale=1):
            with gr.Column(elem_classes="output-container"):
                gr.Markdown("### Analysis Results")
                label_output = gr.Label(
                    label="Prediction & Confidence",
                    show_label=False
                )
    
    gr.Examples(
        examples=[
            # Add your own example images here if available
        ],
        inputs=[image_input]
    )
    
    gr.Markdown(
        """
        <div style='text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.7); border-radius: 8px;'>
            <p style='color: #4a5568; font-size: 0.9rem;'>Developed by Yassin Samy</p>
            <p style='color: #718096; font-size: 0.8rem;'>Using Deep Learning for Plant Disease Detection</p>
        </div>
        """
    )

    submit_btn.click(
        predict_plant,
        inputs=image_input,
        outputs=label_output
    )

    demo.launch()
