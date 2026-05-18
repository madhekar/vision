import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from PIL import Image
import numpy as np

# 1. Initialize Models & ChromaDB
# Bi-encoder model for fast image/text embedding (using CLIP here for multimodal support)
embedding_model = SentenceTransformer("clip-ViT-B-32")

# Cross-encoder for precise reranking 
# (You can use a cross-encoder trained on image-text tasks or text if your query is text-based)
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = chromadb.Client()
collection = client.get_or_create_collection(name="image_store")

# 2. Add Images to ChromaDB 
# (Assuming you already converted your images to vector embeddings)
# image_embeddings = embedding_model.encode([Image.open("img1.jpg"), Image.open("img2.jpg")]).tolist()
# collection.add(
#     embeddings=image_embeddings,
#     uris=["img1.jpg", "img2.jpg"], # Storing file paths
#     ids=["id1", "id2"]
# )

# 3. Query Phase - Stage 1: Fast Retrieval (Bi-Encoder)
query_image = Image.open("query_image.jpg")
query_embedding = embedding_model.encode(query_image).tolist()

# Retrieve top 20 candidate images from ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=20,
    include=["uris", "documents", "embeddings"]
)

# Extract candidates
candidate_uris = results["uris"][0]
candidate_embeddings = results["embeddings"][0]

# 4. Query Phase - Stage 2: Deep Reranking (Cross-Encoder)
# Create pairs: [Query Image, Candidate Image] for the cross-encoder to score
pairs = []
for uri in candidate_uris:
    # Load candidate image to pair with the query image
    candidate_img = Image.open(uri) 
    pairs.append([query_image, candidate_img])

# Predict relevance scores
scores = reranker_model.predict(pairs)

# 5. Sort and Display Top K Results
# Combine URIs and their scores, then sort by relevance
scored_results = list(zip(candidate_uris, scores))
reranked_results = sorted(scored_results, key=lambda x: x[1], reverse=True)

# Print top 5 reranked images
top_k = 5
for i, (uri, score) in enumerate(reranked_results[:top_k]):
    print(f"Rank {i+1} | Image: {uri} | Cross-Encoder Score: {score:.4f}")
