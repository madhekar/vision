import clip
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import util

# paths
#train_json_path = "/home/madhekar/work/zsource/family/metadata/project-zesha-class.json"
#train_image_path = "/home/madhekar/work/zsource/family/img_train/"
subclasses = util.subclasses



def inferring_from_model(model_ft, file_name):

    # Transform the image
    image_tensor = (
        transform(Image.open(file_name)).unsqueeze(0).to(device)
    )  # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        output = model_ft(image_tensor)
        _, predicted_label_idx = torch.max(output, 1)
        predicted_label = subclasses[predicted_label_idx.item()]

    st.write(predicted_label)     


if __name__ == "__main__":
  
  st.title("* OpenCLIP FineTune *")
  # Load the CLIP model architecture
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

  # Preprocess the image
  transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ]
  )
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

  num_classes = len(subclasses)
  model_ft = CLIPFineTuner(model, num_classes).to(device)

  # Load the saved model weights
  model_ft.load_state_dict(torch.load("clip_finetuned.pth"))
  model_ft.eval()  # Set the model to evaluation mode

  select_file = st.file_uploader('select image file', accept_multiple_files=False)
  if select_file:
     im = Image.open(select_file)
     name = select_file.name
     st.image(im, caption="selected image to find simili...")

  btn = st.button('predict caption')
  if btn: 
    inferring_from_model(model_ft, select_file)