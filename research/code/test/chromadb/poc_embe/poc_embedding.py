import torch
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Load the unified CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 2. Supply your custom image (Downloading a sample cat photo)
# img_url = "https://unsplash.com"
image = Image.open("/mnt/zmdata/home-media-app/data/final-data/img/SWEETHOME/c5e0aeb4-10cb-50e6-a19c-e15c3c0235ed/fce7616e-d485-403b-ba29-e33d5b80df09-1.jpg") #requests.get(img_url, stream=True).raw)

# 3. Supply your custom text
#text_prompt = "four people group taking selfie in a farmland in India. two women and two men wearing glasses are warm smiling, towards the left side of them an old hut is visible in background by the left side of a medium size tree."

text_prompt = "The image captures a moment shared by four individuals. Kumar, with his glasses and red shirt, stands alongside Asha, who is wearing a white saree. Two women are also present in the picture; one of them can be seen holding a purse. They all appear to be posing for the photo with cheerful expressions on their faces. The setting seems serene, surrounded by nature, suggesting that they might be enjoying a day out or celebrating an occasion at this location."
# 4. Preprocess both inputs for the model
inputs = processor(text=[text_prompt], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)

# 5. Extract the embeddings
with torch.no_grad():
    # Forward pass
    outputs = model(**inputs)

    # Extract and normalize
    image_features = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_features = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

    """     # Generate raw features
    image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
    text_features = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    # Normalize vectors (Essential step to put them on a comparable scale)
    image_features = model.get_image_features(**inputs)[0] 
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True) """

# Convert to NumPy for clean printing
img_vector = image_features.squeeze().numpy()
txt_vector = text_features.squeeze().numpy()

# 6. Print the results
print("=" * 60)
print(f"IMAGE EMBEDDING (Shape: {img_vector.shape})")
print(f"First 5 dimensions: {img_vector[:5]}")
print("-" * 60)
print(f"TEXT EMBEDDING (Shape: {txt_vector.shape})")
print(f"First 5 dimensions: {txt_vector[:5]}")
print("=" * 60)

# 7. Calculate Similarity Score (Dot product of normalized vectors)
similarity = (image_features @ text_features.T).item()
print(f"Cosine Similarity Score: {similarity:.4f}")
print(f"Match Probability: {similarity * 100:.2f}%")
print("=" * 60)
