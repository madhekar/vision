from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

# 1. Define the augmentation pipeline
# Transforms are chained using Compose
# The order matters, especially for operations like ToTensor()
data_augmentations = transforms.Compose([
    transforms.RandomResizedCrop(224), # Crops and resizes to 224x224
    transforms.RandomHorizontalFlip(p=0.5), # Randomly flips with 50% chance
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Randomly changes color properties
    transforms.ToTensor(), # Converts PIL Image or NumPy array to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizes tensor image
])

# 2. Load an image using PIL
img_path = "/home/madhekar/tmp/esha1.png"
try:
    pil_image = Image.open(img_path).convert('RGB')
except FileNotFoundError:
    print(f"Error: {img_path} not found. Please provide a valid image path.")
    # You might want to use a placeholder image or exit here
    pil_image = Image.new('RGB', (256, 256), color = 'red') # Example placeholder

# 3. Apply the transformations
# The augmentation happens dynamically when the transform object is called on the image
augmented_tensor = data_augmentations(pil_image)

# 4. (Optional) Convert back to PIL for visualization
# Need to permute channels from C, H, W to H, W, C for matplotlib
# and un-normalize the image
def tensor_to_pil(tensor_img):
    tensor_img = tensor_img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    tensor_img = torch.clamp(tensor_img, 0, 1)
    return transforms.ToPILImage()(tensor_img)

# Show original and an augmented image

plt.imshow(pil_image)
plt.title("Original Image")
plt.show()

plt.imshow(tensor_to_pil(augmented_tensor))
plt.title("Augmented Image")
plt.show()