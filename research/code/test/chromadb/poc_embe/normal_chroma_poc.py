import pandas as pd
import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


image_description = [
 {
     'id': 'img_001',
     'img_path' : "/mnt/zmdata/home-media-app/data/final-data/img/SWEETHOME/c5e0aeb4-10cb-50e6-a19c-e15c3c0235ed/fce7616e-d485-403b-ba29-e33d5b80df09-1.jpg",
     "description" : "The image captures a moment shared by four individuals. Kumar, with his glasses and red shirt, stands alongside Asha, who is wearing a white saree. Two women are also present in the picture; one of them can be seen holding a purse. They all appear to be posing for the photo with cheerful expressions on their faces. The setting seems serene, surrounded by nature, suggesting that they might be enjoying a day out or celebrating an occasion at this location."
 },
 {
     'id': 'img_002',
     'img_path' : "/mnt/zmdata/home-media-app/data/final-data/img/SWEETHOME/c5e0aeb4-10cb-50e6-a19c-e15c3c0235ed/IMG_3120-1.jpg",
     "description" : "In the image, there is a delightful beach scene featuring two young women standing on the sandy shore with the ocean visible behind them. The woman on the left, identified as Esha, is smiling broadly and appears to be enjoying her time at the beach, embodying a happy soul. Her companion is also smiling, indicating they are both having a pleasant experience.\n\nThey are positioned close together, suggesting a friendly relationship or camaraderie between them. The ocean in the background has calm waters with small waves, providing a serene and picturesque setting for their outing.\n\nThe reference to \"Madhekar residence in Carmel Valley\" might imply that this beach is located near or within the vicinity of such a residential area. However, it's important to note that the image does not provide direct evidence of the specific location being Madhekar's residence in Carmel Valley, and it could be merely an assumption based on the limited context provided."
 },
  {
     'id': 'img_003',
     'img_path' : "/mnt/zmdata/home-media-app/data/final-data/img/SWEETHOME/c5e0aeb4-10cb-50e6-a19c-e15c3c0235ed/IMG_7064-1.JPG",
     "description" : "The image features a young girl named Esha sitting at a table inside what appears to be the Madhekar Residence Home located in San Diego. She is wearing a blue top and seems to be in a joyful mood, as indicated by her bright smile. Her surroundings suggest an indoor setting with kitchen appliances visible in the background. The image conveys a sense of warmth and happiness associated with Esha's character, as well as the inviting ambiance of her home environment."
  },
  {
     'id': 'img_004',
     'img_path' : "/mnt/zmdata/home-media-app/data/final-data/img/SWEETHOME/c5e0aeb4-10cb-50e6-a19c-e15c3c0235ed/IMG_8233-1.JPG",
     "description" : "The image captures a delightful moment at the Madhekar residence in Carmel Valley. There are four individuals present - Esha, Anjali, another girl, and one more person whose name is not mentioned. They are all facing towards the camera, their smiles radiant as they look up. \n\nEsha, wearing a patterned top, stands to the left of the frame, her smile warm and inviting. In the center of the image, Anjali can be seen in a white shirt. She too is smiling broadly, adding to the cheerful atmosphere. To the right of the frame is another girl; she's also looking up at the camera with a big smile on her face.\n\nThe fourth person is located in the bottom left corner of the image. They are wearing glasses and are also smiling as they look towards the camera. The Madhekar residence, visible in the background, provides a homely backdrop to this cheerful gathering."
 },
  {
     'id': 'img_005',
     'img_path' : "/mnt/zmdata/home-media-app/data/final-data/img/SWEETHOME/c5e0aeb4-10cb-50e6-a19c-e15c3c0235ed/IMG_7060-1.JPG",
     "description" : "The image depicts a warm and intimate scene at the Madhekar residence in Carmel Valley. In the center of the frame, Esha, a happy soul, is seated on the floor with her legs stretched out in front of her. She's wearing a comfortable blue shirt that matches the lively energy she exudes.\n\nTo her right, another girl is sitting down as well, engaged in conversation or perhaps enjoying some quiet time together. The two girls seem to be having a pleasant interaction, contributing to the overall relaxed atmosphere of the scene.\n\nOn Esha's left, another person, Esha - the neutral soul, stands with an air of calmness and composure that contrasts with her twin's effervescent mood. She is dressed in a blue top as well, creating a sense of harmony within the image.\n\nThe kitchen counter in the background is cluttered with various items, including a cup and some food wrappers, indicating recent activity or ongoing preparation for a meal. The presence of these everyday objects adds a layer of authenticity to this snapshot of life at the Madhekar residence."
  }
]

df = pd.DataFrame(image_description)

# Initialize ChromaDB client (local persistent or in-memory)
client = chromadb.PersistentClient(path="./chroma_db_normal")

# Setup data loader and multimodal embedding function
data_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()

# Create or get the collection
collection = client.get_or_create_collection(
    name="multimodal_collection", 
    embedding_function=embedding_function, 
    data_loader=data_loader
)

desc = [{"description": row} for row in df["description"].tolist()]

print(desc)
# Pass unique IDs and the actual URIs (file paths) of your images
collection.add(
    ids=df["id"].tolist(),
    uris=df["img_path"].tolist(),
    metadatas=desc
)

results = collection.query(
    query_texts=["Anjali"],
    n_results=1,
    include=["uris", "distances", "metadatas"] # include URIs to fetch image file paths
)

print(results) 
# Output will contain the URIs matching the query, e.g., [['path/to/tiger.jpg', 'path/to/lion.jpg']]
