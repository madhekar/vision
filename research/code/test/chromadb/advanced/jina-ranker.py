import os
import requests
from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# 1. Initialize Chroma Client & Embedding Function
# OpenCLIP is commonly used for multimodal embedding inside Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = OpenCLIPEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="multimodal_collection", 
    embedding_function=embedding_function
)

# 2. Populate Chroma with Document Images (Initial Setup)
# Ensure your images exist at these paths
image_paths = ["page1.png", "page2.png", "page3.png"]

collection.add(
    ids=[f"img_{i}" for i in range(len(image_paths))],
    uris=image_paths,  # Stores the visual link
    metadatas=[{"file_path": path} for path in image_paths]
)

# 3. Stage 1: Vector Search in Chroma (Retrieve Top K Candidates)
user_query = "Find the layout chart showcasing revenue growth"
initial_results = collection.query(
    query_texts=[user_query],
    n_results=3  # Get top candidates to narrow down
)

# Extract image file paths from our initial Chroma vector match
candidate_paths = [meta["file_path"] for meta in initial_results["metadatas"][0]]

# 4. Stage 2: Rerank Candidates Using jina-reranker-m0 via API
# Get your API key from https://jina.ai/reranker/
JINA_API_KEY = os.environ.get("JINA_API_KEY", "your_jina_api_key_here")
url = "https://jina.ai"

headers = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json"
}

# Construct the multimodal payload according to the Jina API specification
# In a production app, convert local images to Base64 strings or public URLs
documents = []
for path in candidate_paths:
    # Example template assuming your images are accessible or converted to base64
    documents.append({
        "text": f"Document snippet or metadata for {path}", 
        "image": path  # Note: Jina API expects a public URL or a base64 encoded data URI
    })

payload = {
    "model": "jina-reranker-m0",
    "query": user_query,
    "documents": documents,
    "top_n": 2
}

response = requests.post(url, headers=headers, json=payload)
reranked_data = response.json()

# 5. Output the accurately reordered search results
print("Reranked Results:")
for result in reranked_data.get("results", []):
    idx = result["index"]
    score = result["relevance_score"]
    print(f"File: {candidate_paths[idx]} | Relevance Score: {score:.4f}")
