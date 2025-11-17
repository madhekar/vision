import os
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from utils.config_util import config



# Define directories, files, params
def load_init_params():
    (
        filter_model_path,
        data_path,
        filter_model_name,
        filter_model_classes,
        image_size,
        batch_size,
        epocs
    ) = config.filer_config_load()

    batch_size_int = int(batch_size)
    sz = ast.literal_eval(image_size)
    epocs_int = int(epocs)
    image_size_int = (int(sz[0]), int(sz[1]))
    return filter_model_path, data_path, filter_model_name, filter_model_classes, image_size_int, batch_size_int, epocs_int

def torch_model(data_dir_path, filter_model_path, filter_model_name, filter_model_classes, image_size_int, batch_size_int, device, num_epochs):

    data_transforms = {
        "training": transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size_int),
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
                transforms.CenterCrop(image_size_int),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir_path, x), data_transforms[x])
        for x in ["training", "validation"]
    }
    print(image_datasets)
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size_int, shuffle=True, num_workers=8)
        for x in ["training", "validation"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["training", "validation"]}
    class_names = image_datasets["training"].classes
    class_mappings = image_datasets["training"].class_to_idx
    print(f"num classes: {class_names} class to idx: {class_mappings}")

    # model_ft = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    # Or for MobileNetV3:
    model_ft = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)

    # model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model_ft.to(device)

    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1] = nn.Linear(num_ftrs, len(class_names)).to(device)

    # num_classes = 3  # Replace with your actual number of classes
    # model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes).to(device)

    # for param in model_ft.features.parameters():
    #     param.requires_grad = False

    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00001)
    optimizer_ft = optim.Adam(
        model_ft.parameters(), lr=0.0001, betas=(0.9, 0.99), eps=1e-08
    )

    # Example traininging loop snippet
    for epoch in range(num_epochs):
        for phase in ["training", "validation"]:
            if phase == "training":
                model_ft.train()
            else:
                model_ft.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(
                    device
                )  # Assuming device is defined (e.g., 'cuda' or 'cpu')
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

            print(f"{epoch}:{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # save model
    print(f"saving model {filter_model_name} at {filter_model_path}...")
    torch.save(model_ft, os.path.join(filter_model_path, filter_model_name)) 

    # save mapping labels
    print(f"saving mappings {filter_model_classes} at {filter_model_path}")
    torch.save(class_mappings, os.path.join(filter_model_path, filter_model_classes)) 


def execute():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    (
        filter_model_path,
        data_path,
        filter_model_name,
        filter_model_classes,
        image_size_int,
        batch_size_int,
        num_epochs
    ) = load_init_params()

    torch_model(
        data_path,
        filter_model_path,
        filter_model_name,
        filter_model_classes,
        image_size_int,
        batch_size_int,
        device,
        num_epochs
    )