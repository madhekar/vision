import torch
import pyiqa
from PIL import Image
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nima_metric = pyiqa.create_metric('nima', device=device)

# image preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

sample_img = (
    "/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup/images.jpeg"
)

image = Image.open(sample_img)
image_tensor = preprocess(image).unsqueeze(0)

normalized_tensor = (image_tensor - image_tensor.min()) / (
    image_tensor.max() - image_tensor.min()
)
with torch.no_grad():
    score = nima_metric(normalized_tensor)

print(f'nima score: {score.item():.4f}')    