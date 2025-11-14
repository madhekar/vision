import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

num_epochs = 50
# 1. Load the pre-trained MobileNetV3_Large model
# You can choose 'mobilenet_v3_small' if needed
#model_ft = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

#model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

model_ft = models.alexnet(pretrained=True)
model_ft.classifier[6] = torch.nn.Linear(model_ft.classifier[6].in_features, 3)

# 2. Modify the classifier for your specific task
# MobileNetV3's classifier is a sequential block. The last layer is the one we replace.
# The `in_features` for the new linear layer needs to match the `out_features` of the layer before it.

# num_ftrs = model_ft.classifier[-1].in_features
# num_classes = 3 # Replace with the actual number of classes in your dataset
# model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)


#num_classes = 3 # Replace with your actual number of classes
#model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)

# 3. (Optional) Freeze earlier layers
# This is common in fine-tuning to prevent overfitting and speed up training,
# especially when your dataset is small.
# for param in model_ft.features.parameters():
#     param.requires_grad = False

# 4. Define your data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 5. Create a custom dataset (example using a dummy dataset structure)
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # Assuming subfolders in root_dir are class names
        for i, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        #print(f"--->img: {img_path} lbl: {label}")
        if self.transform:
            image = self.transform(image)
        return image, label

# Create dummy data directories for demonstration
# In a real scenario, replace 'data/train' and 'data/val' with your actual paths
# os.makedirs('data/train/class1', exist_ok=True)
# os.makedirs('data/train/class2', exist_ok=True)
# os.makedirs('data/val/class1', exist_ok=True)
# os.makedirs('data/val/class2', exist_ok=True)

# Example usage (replace with your actual data loading)
image_datasets = {
    'train': CustomDataset(root_dir='/home/madhekar/temp/filter/training', transform=data_transforms['train']),
    'val': CustomDataset(root_dir='/home/madhekar/temp/filter/validation', transform=data_transforms['val'])
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4)
    for x in ['train', 'val']
}
#print(image_datasets)
# 6. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.parameters(), lr=0.001) # Or selectively optimize only the new layer

# 7. Training loop (conceptual, replace with your actual training logic)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

for epoch in range(num_epochs):
    # Training phase
    epoch_loss = 0.0
    running_loss = 0.0
    model_ft.train()
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_ft(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / 16   
    print(f"epoch {epoch} / {num_epochs} loss: {epoch_loss}")

    # Validation phase
    model_ft.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_ft(inputs)
            # Calculate validation metrics
            loss = criterion(outputs, labels)
