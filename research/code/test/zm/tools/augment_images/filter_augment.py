import os
import torch
from torchvision import transforms
from PIL import Image

# Define Augmentation Pipeline
# Use v2 transforms for better performance (requires torchvision >= 0.15)
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0), # Flip for example
    transforms.RandomRotation(degrees=30),  # Rotate for example
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Brightness shift
    transforms.RandomResizedCrop(224),
    transforms.RandomAutocontrast(0.5),
    transforms.RandomAdjustSharpness(0.5)
])

def apply_aug(filters_folder):
    for filter in os.listdir(filters_folder):
        for filename in os.listdir(os.path.join(filters_folder, filter)): 
            if filename.endswith(".jpg") or filename.endswith(".png"):                    
                # Avoid augmenting already augmented images
                if filename.startswith("aug_"):
                    continue
                for i in range(10):
                    file_path = os.path.join(filters_folder, filter, filename)
                    img = Image.open(file_path).convert('RGB')
                    
                    # Apply Augmentation
                    augmented_img = augmentation(img)
                    
                    # Save in Same Folder
                    save_path = os.path.join(filters_folder, filter, f"aug_{i}_{filename}")
                    augmented_img.save(save_path)
                    print(f"Saved: {save_path}")

    print("Augmentation Complete.")


if __name__=="__main__":
    filters_root = "/mnt/zmdata/home-media-app/data/app-data/static-metadata/filter/training"
    apply_aug(filters_root)