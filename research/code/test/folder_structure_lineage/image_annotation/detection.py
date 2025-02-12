import torch
from torchvision import transforms
from PIL import Image

# Define the transformations
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load an image (replace with your image loading method)
image = Image.open(
    "/home/madhekar/work/home-media-app/data/input-data/img/Coach Eval Sem2 002.png"#IMAG2294.jpg"#imgIMG_4776.jpeg"
).convert("RGB")

# Apply the transformations correctly
transformed_image = transform(image)

# transformed_image is now the transformed tensor
print(transformed_image.shape)
