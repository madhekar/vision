import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Load the unified CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

image = Image.open("/mnt/zmdata/home-media-app/data/final-data/img/SWEETHOME/c5e0aeb4-10cb-50e6-a19c-e15c3c0235ed/fce7616e-d485-403b-ba29-e33d5b80df09-1.jpg")
text_prompt = "The image captures a moment shared by four individuals. Kumar, with his glasses and red shirt, stands alongside Asha, who is wearing a white saree. Two women are also present in the picture; one of them can be seen holding a purse. They all appear to be posing for the photo with cheerful expressions on their faces. The setting seems serene, surrounded by nature, suggesting that they might be enjoying a day out or celebrating an occasion at this location."

# Initialize standard client
client = chromadb.PersistentClient(path="./my_chromadb_embedding")
# Do NOT pass an embedding_function here; we will handle it ourselves
collection = client.get_or_create_collection(name="manual_clip_collection")

# --- FROM PREVIOUS STEP ---
# Let's say you already ran your code and extracted:
# img_vector = [0.024, -0.011, ...]  (Length 512 list or numpy array)
# txt_vector = [0.021, -0.015, ...]  (Length 512 list or numpy array)

inputs = processor(text=[text_prompt], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)

# 5. Extract the embeddings
with torch.no_grad():
    # Forward pass
    outputs = model(**inputs)

    # Extract and normalize
    image_features = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_features = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

# Convert to NumPy for clean printing
img_vector = image_features.squeeze().numpy()
txt_vector = text_features.squeeze().numpy()

#print(img_vector.tolist())
# 1. Insert raw image embeddings manually
collection.add(
    ids=["img_id_001"],
    embeddings=[img_vector.tolist()], # Must be converted to a Python list
    metadatas=[{"source_path": "/mnt/zmdata/home-media-app/data/final-data/img/SWEETHOME/c5e0aeb4-10cb-50e6-a19c-e15c3c0235ed/fce7616e-d485-403b-ba29-e33d5b80df09-1.jpg"}],
    documents=["The image captures a moment shared by four individuals. Kumar, with his glasses and red shirt, stands alongside Asha, who is wearing a white saree. Two women are also present in the picture; one of them can be seen holding a purse. They all appear to be posing for the photo with cheerful expressions on their faces. The setting seems serene, surrounded by nature, suggesting that they might be enjoying a day out or celebrating an occasion at this location."
]
)

# 2. Query using a raw text embedding vector manually
query_results = collection.query(
    query_embeddings=[txt_vector.tolist()], # Pass the text vector directly
    n_results=1
)
print(query_results)
