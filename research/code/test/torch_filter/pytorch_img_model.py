import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 50

data_transforms = {
    "training": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=30),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "validation": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

data_dir = '/home/madhekar/temp/filter' # Replace with your dataset path
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['training', 'validation']}
print(image_datasets)
dataloaders = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=8) for x in ['training', 'validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['training', 'validation']}
class_names = image_datasets['training'].classes
class_mappings =  image_datasets['training'].class_to_idx
print(f"num classes: {class_names} class to idx: {class_mappings}")

#model_ft = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
# Or for MobileNetV3:
model_ft = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

# model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model_ft.to(device)

num_ftrs = model_ft.classifier[-1].in_features
model_ft.classifier[-1] = nn.Linear(num_ftrs, len(class_names)).to(device)

# num_classes = 3  # Replace with your actual number of classes
# model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes).to(device)

# for param in model_ft.features.parameters():
#     param.requires_grad = False

criterion = nn.CrossEntropyLoss().to(device)
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00001)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001, betas=(0.9, 0.99), eps=1e-08)

# Example traininging loop snippet
for epoch in range(num_epochs):
    for phase in ['training', 'validation']:
        if phase == 'training':
            model_ft.train()
        else:
            model_ft.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device) # Assuming device is defined (e.g., 'cuda' or 'cpu')
            labels = labels.to(device)

            optimizer_ft.zero_grad()

            with torch.set_grad_enabled(phase == "training"):
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "training":
                    loss.backward()
                    optimizer_ft.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# save model
torch.save(model_ft, "filter_model.pth")

# save mapping labels
torch.save(class_mappings, "label_mappings.pth")

# filter_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
# filter_model.load_state_dict(torch.load("filter_model_weights.pth"))

# load model
filter_model = torch.load("filter_model.pth", weights_only=False)

# load mapping labels
class_map = torch.load("label_mappings.pth")

print(class_mappings)
print(filter_model)

filter_model.eval()