import torch
import torchvision
import urllib
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_image_channels_from_url(url):
    try:
        img_pil = Image.open(url)#.convert('RGB')
        img_pil.mode, img_pil.size
        return img_pil, img_pil.mode, img_pil.size
    except Exception as e:
        print(f"Error: Could not process image from URL. {e}")
        return None

def tx(url):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.48145466, 0.4578275, 0.40821073],
            #     std=[0.26862954, 0.26130258, 0.27577711],
            # ),
        ]
    )

    image_tensor = (transform(Image.open(url)).unsqueeze(0).to(device))  # Add batch dimension and move to device

    image_tensor(img)

# Example usage
url = "/home/madhekar/work/home-media-app/data/input-data/img/Coach Eval Sem2 002.png"
img, mode, size = get_image_channels_from_url(url)

#tx(url)

print(f"The image has mode: {mode} size: {size}.")
