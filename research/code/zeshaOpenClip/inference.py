import clip
import torch
import os
import json
from PIL import Image

# Load the CLIP model architecture
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# paths
train_json_path = "/home/madhekar/work/zsource/family/metadata/project-zesha-class.json"
train_image_path = "/home/madhekar/work/zsource/family/img_train/"
test_image_path = "/home/madhekar/work/zsource/family/img/"

# obect names
object_names = [
    "Esha",
    "Anjali",
    "Bhalchandra",
    "Esha.Anjali",
    "Esha,Bhalchandra",
    "Esha,Anjali,Bhalchandra",
    "Esha,Shibangi",
    "Anjali,Shoma",
    "Esha,Asha",
    "Shoma",
    "Bhiman",
]

# input data as a list
input_data = []
with open(train_json_path, "r") as file:
    for line in file:
        obj = json.loads(line)
        input_data.append(obj)

# Load the saved state dictionary
model.load_state_dict(torch.load("./trained_clip_model.pth"))


# Set the model to evaluation mode
model.eval()

#
def inferring_from_model(sample_index):
    image_json = input_data[sample_index]

    # Construct the full path to the image file using the given 'image_path'
    image_path = os.path.join(test_image_path, image_json["file_name"])
    """  image_class = image_json["class"]
    image_path, image_class """
    print("Image path: ", image_path)
    
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    #print(image.shape)
    
    # Tokenize and move the people item names to the appropriate device
    text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in object_names]).to(
    device)
    #print(text.shape)

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

inferring_from_model(sample_index=2)