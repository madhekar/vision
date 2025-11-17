import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from utils.config_util import config
from PIL import Image



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

'''
test model
'''
def prep_test_data(test_data_root):
    test_data = []
    try:
        entries = os.listdir(test_data_root)
        print(entries)
        for entry in entries:
            print(os.path.join(test_data_root, entry))
            loc_path = os.path.join(test_data_root, entry)
            print("loc", loc_path)
            if os.path.isdir(loc_path):
                print("--->", entry)
                type = entry
                print()
                files = os.listdir(loc_path)
                print(files)
                for file in files:
                    img_file = os.path.join(loc_path, file)
                    class_name = type
                    test_data.append([img_file, class_name])
    except Exception as e:
        print(f"failed with : {e}")
    return test_data


def load_filter_model(fmp, fmn, fmc,isz, device):
    # 1. Load a pre-trained model (e.g., ResNet18)
    # For a custom model, you would define your model architecture and load its state_dict
    filter_model = torch.load(os.path.join(fmp, fmn), weights_only=False)
    filter_model.to(device)
    class_mapping = torch.load(os.path.join(fmp, fmc))
    class_mapping = {v: k for k, v in class_mapping.items()}
    # class_mapping =  ast.literal_eval(class_mapping)
    print(class_mapping)
    print(filter_model)

    filter_model.eval()  # Set the model to evaluation mode

    # 2. Define image transformations
    # These should match the transformations used during training
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(isz),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return filter_model, preprocess, class_mapping

def test_model(img_path, class_name, class_map, filter_model, preprocess, device):

    # 3. Load and preprocess the image
    # Replace 'path/to/your/image.jpg' with the actual path to your image
    input_image = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model

    # 4. Make a prediction
    with torch.no_grad(): # Disable gradient calculation for inference
        output = filter_model(input_batch.to(device))

    # 5. Interpret the output
    # For ImageNet, there are 1000 classes.
    # The output tensor contains raw scores (logits) for each class.
    probabilities = F.softmax(output, dim=1) # Convert logits to probabilities
    predicted_probability, predicted_class_idx = torch.max(probabilities, 1)

    # Optional: Load class labels for better interpretation
    # You would need a list or dictionary mapping class indices to names
    # For ImageNet, you can get the class labels from a file or a pre-defined list
    # Example:
    # with open("imagenet_classes.txt", "r") as f:
    #     imagenet_classes = [line.strip() for line in f.readlines()]
    predicted_class_name = class_map[predicted_class_idx.item()]

    print(f"actual -> class: {class_name} predicted -> class: {predicted_class_name} prob: {predicted_probability.item():.4f}")

    return (predicted_class_name, class_name)

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

    fm, pp, cm = load_filter_model(filter_model_path, filter_model_name, filter_model_classes, image_size_int, device)

    test_data = prep_test_data(os.path.join(data_path, "testing"))

    res = [test_model(e[0], e[1], cm, fm, pp, device) for e in test_data]
    ys, yd = map(list, zip(*res))
    return ys, yd