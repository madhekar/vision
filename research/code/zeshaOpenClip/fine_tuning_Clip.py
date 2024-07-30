
import clip
#from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import json
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

'''
load zesha image data
'''
# paths
train_json_path = "/home/madhekar/work/zsource/family/metadata/project-zesha-class.json"
train_image_path = "/home/madhekar/work/zsource/family/img_train/"

# input data as a list
ds = []
with open(train_json_path, "r") as file:
    for line in file:
        obj = json.loads(line)
        ds.append(obj)

list_image_path = []
list_txt = []
for item in ds:
    img_path = train_image_path + item["file_name"]
    list_image_path.append(img_path)

    # As we have image text pair, we use product title as description.
    caption = item["text"]
    list_txt.append(caption)

'''
load clip model and preprocessing
'''

device = "cuda" if torch.cuda.is_available() else "cpu"

# pretrained-model
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

'''
validate
'''
indices = [3, 4, 5]
subclasses = list(set(example['class'] for example in ds))

# Preprocess the text descriptions for each subcategory
text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in subclasses]).to(device)

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loop through the indices and process each image
for i, idx in enumerate(indices):
    # Select an example image from the dataset
    example = ds[idx]
    image = train_image_path + example["file_name"]
    subclass = example["class"]

    # Preprocess the image
    image_input = preprocess(Image.open(image)).unsqueeze(0).to(device)

    # Calculate image and text features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Normalize the features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity between image and text features
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, inx = similarity[0].topk(1)
    # Display the image in the subplot
    axes[i].imshow(Image.open(image))
    axes[i].set_title(f"Predicted: {subclasses[inx[0]]}, Actual: {subclass}")
    axes[i].axis("off")

# Show the plot
plt.tight_layout()
plt.show()

'''
custom class
'''

# Define a custom dataset class
class ZPhotoTitleDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.ColorJitter(brightness=0.5, hue=0.1, saturation=0.05),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = train_image_path + item["file_name"]
        subclass = item["class"]
        label = subclasses.index(subclass)
        return self.transform(Image.open(image)), label
    
  
zdataset = ZPhotoTitleDataset(ds)
train_dataloader = DataLoader(zdataset, batch_size=16, shuffle=True)

'''
modify model for fine tuning
'''
class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()  # Convert to float32
        return self.classifier(features)

num_classes = len(subclasses)
model_ft = CLIPFineTuner(model, num_classes).to(device)

'''
Define the loss function and optimizer
'''    
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)

'''

'''
# Number of epochs for training
num_epochs = 200

# Training loop
for epoch in range(num_epochs):
    model_ft.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize running loss for the current epoch
    pbar = tqdm(
        train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: 0.0000"
    )  # Initialize progress bar

    for images, labels in pbar:
        images, labels = (
            images.to(device),
            labels.to(device),
        )  # Move images and labels to the device (GPU or CPU)
        optimizer.zero_grad()  # Clear the gradients of all optimized variables
        outputs = model_ft(
            images
        )  # Forward pass: compute predicted outputs by passing inputs to the model
        loss = criterion(outputs, labels)  # Calculate the loss
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()  # Perform a single optimization step (parameter update)

        running_loss += loss.item()  # Update running loss
        pbar.set_description(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader):.4f}"
        )  # Update progress bar with current loss

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}")  # Print average loss for the epoch

    # Validation
    model_ft.eval()  # Set the model to evaluation mode
    correct = 0  # Initialize correct predictions counter
    total = 0  # Initialize total samples counter

    with torch.no_grad():  # Disable gradient calculation for validation
        for images, labels in train_dataloader:
            images, labels = (
                images.to(device),
                labels.to(device),
            )  # Move images and labels to the device
            outputs = model_ft(
                images
            )  # Forward pass: compute predicted outputs by passing inputs to the model
            _, predicted = torch.max(
                outputs.data, 1
            )  # Get the class label with the highest probability
            total += labels.size(0)  # Update total samples
            correct += (predicted == labels).sum().item()  # Update correct predictions

    print(f"Validation Accuracy: {100 * correct / total}%")  # Print validation accuracy for the epoch

# Save the fine-tuned model
torch.save(
    model_ft.state_dict(), "clip_finetuned.pth"
)  # Save the model's state dictionary

'''

'''
# Load the saved model weights
model_ft.load_state_dict(torch.load("clip_finetuned.pth"))
model_ft.eval()  # Set the model to evaluation mode

# Define the indices for the three images
indices = [3, 4, 5]

# Preprocess the image
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loop through the indices and process each image
for i, idx in enumerate(indices):
    # Get the image and label from the dataset
    item = ds[idx]
    image = train_image_path + item["file_name"]
    true_label = item["class"]

    # Transform the image
    image_tensor = (
        transform(Image.open(image)).unsqueeze(0).to(device)
    )  # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        output = model_ft(image_tensor)
        _, predicted_label_idx = torch.max(output, 1)
        predicted_label = subclasses[predicted_label_idx.item()]

    # Display the image in the subplot
    axes[i].imshow(Image.open(image))
    axes[i].set_title(f"True label: {true_label}\nPredicted label: {predicted_label}")
    axes[i].axis("off")

# Show the plot
plt.tight_layout()
plt.show()