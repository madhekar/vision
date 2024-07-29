import clip

import os
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from util import getNamesCombination
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# pretrained-model
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
# model is for images
# preprocess is for texts


# paths
train_json_path = "/home/madhekar/work/zsource/family/metadata/project-zesha-class.json"

# input data as a list
input_data = []
with open(train_json_path, "r") as file:
    for line in file:
        print(line)
        obj = json.loads(line)
        input_data.append(obj)

#print(input_data[:2])


train_image_path = "/home/madhekar/work/zsource/family/img_train/"

list_image_path = []
list_txt = []

for item in input_data:
    img_path = train_image_path + item["file_name"]
    list_image_path.append(img_path)

    # As we have image text pair, we use product title as description.
    caption = item["text"]
    list_txt.append(caption)

print(list_image_path[:2], list_txt[:2], len(list_image_path))



# Text tokenization is done once at initialization, saving computation during training.
# Images are loaded and preprocessed on demand, saving memory.
# Define a custom dataset
class image_title_dataset:
    def __init__(self, list_image_path, list_txt, xform):
        self.image_path = list_image_path
        self.title = clip.tokenize(list_txt)
        self.xform = xform

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # It loads and preprocesses the image on-the-fly
        # when an item is requested. This is memory-efficient
        # as it doesn't load all images into memory at once
        image = preprocess(Image.open(self.image_path[idx]))
        image = self.xform(image)
        title = self.title[idx]

        # returned is the preprocessed image and the preprocessed (tokenized) title
        return image, title
###
transform = transforms.Compose(
    [
        #transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.5, hue=0.1, saturation=0.05),
        transforms.RandomRotation(20),
        # transforms.Normalize((.5,.5,.5),(.5,.5,.5)),
        transforms.ToTensor(),
    ]
)
###

# to check with only 50 examples
# dataset = image_title_dataset(list_image_path[:10000], list_txt[:10000])

dataset = image_title_dataset(list_image_path, list_txt, transform)

train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

len(next(iter(train_dataloader)))

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

'''
Ground truth and loss calculation:

The ground truth is indeed defined as a simple range of values, which might seem counterintuitive at first. However, this approach is actually implementing a form of contrastive learning. Here's how it works:

    The ground_truth tensor is a range from 0 to batch_size-1.
    logits_per_image and logits_per_text are matrices of size (batch_size, batch_size).
        logits_per_image and logits_per_text are similarity matrices.
        Each row in logits_per_image represents how similar one image is to all the text descriptions in the batch.
        Similarly, each row in logits_per_text represents how similar one text description is to all the images in the batch.
    For each row in these matrices, the correct match should be along the diagonal.

The CrossEntropyLoss is then used to push the model to maximize the similarity between the correct image-text pairs while minimizing the similarity between incorrect pairs. In essence, it's teaching the model to match each image with its corresponding text description within the batch.\ This approach leverages the batch itself as a set of negative examples, which is a clever way to perform contrastive learning without explicitly defining negative pairs.

By using CrossEntropyLoss with this setup, we're essentially saying:

    For each image (or text), make its similarity highest with its corresponding text (or image).
    At the same time, minimize its similarity with all other texts (or images) in the batch.

Positive and Negative Samples:

    Each image-text pair in the batch serves as a positive example for itself.
    All other combinations in the batch serve as negative examples.
    This clever use of the batch as its own set of negative examples is what makes this approach efficient.

This approach allows the model to learn rich, multi-modal representations without needing explicit labels. It can capture nuanced relationships between visual and textual information. The learned representations are often highly transferable to other tasks.

'''

# training
n_epochs = 5

for epoch in range(n_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))

    for batch in pbar:
        optimizer.zero_grad()
        images, texts = batch
        images, texts = images.to(device), texts.to(device)

        # forward pass
        logits_per_image, logits_per_text = model(images, texts)

        # loss is avg of text and image loss
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        total_loss = (
            loss_img(logits_per_image, ground_truth)
            + loss_txt(logits_per_text, ground_truth)
        ) / 2

        total_loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        pbar.set_description(f"Epoch {epoch}/{n_epochs}, Loss: {total_loss.item():.4f}")

#obect names
object_names =  getNamesCombination()

print(len(object_names))

# Index of the input data you want to analyze
index_ = 5

# Assuming 'input_data' is a list of JSON-like objects with image information
image_json = input_data[index_]

# Construct the full path to the image file using the given 'file_name'
image_path = os.path.join(train_image_path, image_json["file_name"])
image_class = image_json["class"]
print(image_path, image_class)


image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
print(image.shape)

# Tokenize and move the people item names to the appropriate device
text = torch.cat([clip.tokenize(f"a image of a {c}") for c in object_names]).to(
    device
)
print(text.shape)


# Perform inference
with torch.no_grad():
    # Encode image and text
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Calculate similarity scores between image and text
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# Normalize image and text features
# normalized to have unit length
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Calculate similarity scores
# dot product measures how similar the image embedding is to each text embedding.
# softmax converts it into probabilities
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the top predictions
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{object_names[index]:>16s}: {100 * value.item():.2f}%")

# Display the image with its class label
plt.imshow(plt.imread(image_path))
plt.title(f"Image for class: {image_class}")
plt.axis("off")
plt.show()

#

#saving the model
torch.save(model.state_dict(), './trained_clip_model.pth')

# Load the CLIP model architecture
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# Load the saved state dictionary
model.load_state_dict(torch.load('./trained_clip_model.pth'))


# Set the model to evaluation mode
model.eval()

#

def inferring_from_model(sample_index):
    image_json = input_data[sample_index]

    # Construct the full path to the image file using the given 'image_path'
    image_path = os.path.join(train_image_path, image_json["file_name"])
    image_class = image_json["class"]
    image_path, image_class
    
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    print(image.shape)

    with torch.no_grad():
        # Encode image and text
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)


    # Normalize image and text features
    # normalized to have unit length
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity scores
    # dot product measures how similar the image embedding is to each text embedding.
    # softmax converts it into probabilities
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    # Print the top predictions
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{object_names[index]:>16s}: {100 * value.item():.2f}%")

    # Display the image with its class label
    plt.imshow(plt.imread(image_path))
    plt.title(f"Image for class: {image_class}")
    plt.axis("off")
    plt.show()


inferring_from_model(sample_index=5)
