
import torch
import torch.nn as nn
import clip
from torchvision import transforms
from PIL import Image
import util
import streamlit as st

# paths
subclasses = util.subclasses

"""
modify model for fine tuning
"""
class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()  # Convert to float32
        return self.classifier(features)

def getEntityNames(image, openclip_finetuned):

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

            # Preprocess the image
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
            )

        num_classes = len(subclasses)
        model_ft = CLIPFineTuner(model, num_classes).to(device)

        # Load the saved model weights
        model_ft.load_state_dict(torch.load( openclip_finetuned))  # "/home/madhekar/work/home-media-app/models/zeshaOpenClip/clip_finetuned.pth"
        model_ft.eval()  # Set the model to evaluation mode

        # Transform the image
        img = Image.open(image).convert("RGB")
        image_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Perform inference
        with torch.no_grad():
            output = model_ft(image_tensor)
            _, predicted_label_idx = torch.max(output, 1)
            predicted_label = subclasses[predicted_label_idx.item()]
    except Exception as e:
        st.error(f"exception occurred in getEntityNames {image} with {e}")    

    return predicted_label
