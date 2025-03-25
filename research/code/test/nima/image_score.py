import nima
import torch
import torchvision.transforms as transforms
from PIL import Image


print(dir(nima))

model = nima(base_model_name='vgg16')
model.load_state_dict(torch.load('nima_vgg16.pth'))
model.eval()

#image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
])

sample_img='/home/madhekar/work/home-media-app/data/train-data/img/AnjaliBackup/IMAG2476.jpg'

image= Image.open(sample_img)
image_tensor = preprocess(image).unsqueeze(0)

with torch.no_grad():
    output =  model(image_tensor)

# calculate the std dev
score_distri = torch.nn.functional.softmax(output, dim=1)    
expected_val = (score_distri * torch.arange(1, 11)).sum(dim=1).item()

# calculate the std dev
std_deviation = torch.sqrt((score_distri * ((torch.arange(1, 11) - expected_val) ** 2)).sum(dim=1)).item()

print(f"NIMA score (mean): {expected_val:.4f}")
print(f"std dev: {std_deviation:.4f}")
