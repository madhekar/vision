import os
import chromadb
from chromadb import Search, K, Knn, Rrf

# 1. Initialize the Chroma Client (Persistent Storage)
# Using a persistent client ensures production readiness.
db_path = "./chroma_production_db"
client = chromadb.PersistentClient(path=db_path)

# 2. Define or Get the Collection
# We'll use the default embedding function for dense vectors
collection_name = "knowledge_base"
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}  # Configure similarity space
)

# 3. Prepare Seed Data (Simulating Dense & Sparse inputs)
# In production, sparse vectors are typically extracted using BM25 or SPLADE
documents = [
    "Deep learning and neural networks advance artificial intelligence research.",
    "A comprehensive guide to machine learning algorithms and statistical models.",
    "Data engineering pipelines and vector databases optimize RAG applications.",
    "Advanced quantum computing architectures and cryptography standards.",
    "Deploying machine learning models to production using Docker and Kubernetes."
]

ids = [f"id_{i}" for i in range(len(documents))]

metadatas = [
    {"status": "published", "category": "AI/ML", "view_count": 1500},
    {"status": "published", "category": "AI/ML", "view_count": 850},
    {"status": "archived", "category": "DataEng", "view_count": 2300},
    {"status": "published", "category": "Quantum", "view_count": 120},
    {"status": "published", "category": "DevOps", "view_count": 3100}
]

# For a true hybrid setup, we store pre-computed sparse array/string mappings in metadata
# Or configure a dedicated sparse vector structure according to your specific cluster setup
mock_sparse_embeddings = [
    [0.1, 0.0, 0.8, 0.0],
    [0.9, 0.1, 0.0, 0.0],
    [0.0, 0.0, 0.2, 0.7],
    [0.0, 0.5, 0.0, 0.1],
    [0.4, 0.0, 0.0, 0.6]
]

# Inject sparse representations directly into the metadata schema for the sparse indexer
for meta, sparse in zip(metadatas, mock_sparse_embeddings):
    meta["sparse_embedding"] = sparse

# Add records to our collection
# Chroma automatically calculates dense embeddings using its native fallback function (e.g., all-MiniLM-L6-v2)
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print(f"Successfully ingested {collection.count()} documents into '{collection_name}'.\n")


# 4. Building the Complex RRF Hybrid Query Pipeline
user_query = "production machine learning research applications"

# Define Dense Semantic Ranker (Knn targets the base vector store)
dense_ranker = Knn(
    query=user_query,
    key="#embedding",       # Maps to the default collection vector field
    return_rank=True,       # Captures rank index position for the fusion step
    limit=100               # Extract a large enough candidate pool for RRF
)

# Define Sparse Keyword Ranker (Knn targets metadata-based sparse mappings)
sparse_ranker = Knn(
    query=user_query,
    key="sparse_embedding", # Maps to custom sparse indicators in the schema
    return_rank=True,
    limit=100
)

# Combine using Reciprocal Rank Fusion (Rrf)
# score = - SUM( weight / (k + rank_position) )
hybrid_ranker = Rrf(
    ranks=[dense_ranker, sparse_ranker],
    weights=[0.7, 0.3],    # Prioritize semantic meaning (70%) over exact token match (30%)
    k=60                   # Standard smoothing constant to avoid over-penalizing tail ranks
)

# 5. Execute Chainable Search with Complex Metadata Expressions
# Only match published documents belonging to specific technical domains
search_pipeline = (
    Search(collection)
    .where(
        (K("status") == "published") & 
        ((K("category") == "AI/ML") | (K("category") == "DevOps"))
    )
    .rank(hybrid_ranker)
    .limit(3)              # Top K items to pass onto your downstream LLM / RAG step
)

# Execute query against your Chroma cluster
results = search_pipeline.execute()

# 6. Parse and Inspect Merged Results
print(f"--- Top Hybrid Search Results (RRF) for: '{user_query}' ---")
for index, hit in enumerate(results):
    # Chroma returns scores ascendingly (lower negative numbers mean higher rank)
    print(f"\n[Rank {index + 1}] ID: {hit.id} | Score: {hit.score:.4f}")
    print(f"Document: {hit.document}")
    print(f"Metadata: Category: {hit.metadata.get('category')} | Status: {hit.metadata.get('status')}")
