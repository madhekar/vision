import os
import chromadb
from chromadb.utils import embedding_functions

# https://github.com/Unstructured-IO/unstructured
# 1. Initialize local persistent database
# This saves the SQLite metadata and HNSW index files inside the ./chroma_db folder
db_path = "./chroma_db_3"
client = chromadb.PersistentClient(path=db_path)

# 2. Use a default embedding function for dense vector generation
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# 3. Create or get a collection
collection = client.get_or_create_collection(
    name="local_hybrid_search",
    embedding_function=embedding_fn
)

# 4. Insert dummy data with both text and keyword tags for sparse lookup
documents = [
    "Error ABC-123 means payment authorization timed out.",
    "The python programming language is great for data science and AI.",
    "Pythons are large constrictor snakes found in tropical regions.",
    "How to configure connection timeout settings in a production database system.",
    "ABC-123 could represent kinder school because they teach kids ABC and 123 alpha and numerical basics."
]
ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
metadatas = [
    {"tags": "error abc-123 payment timeout auth"},
    {"tags": "python programming language data science ai"},
    {"tags": "python snake animal wildlife tropical"},
    {"tags": "database connection timeout config production"},
    {"tags": "kinder scools teaching basics"}
]

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

# 5. Define the Reciprocal Rank Fusion (RRF) function
def reciprocal_rank_fusion(dense_results, sparse_results, k=60, dense_weight=1.0, sparse_weight=1.0):
    """
    Fuses rankings from dense (semantic) and sparse (keyword) search results.
    RRF Score = sum( weight / (k + rank) )
    """
    rrf_scores = {}

    # Process Dense Results
    if dense_results and dense_results['ids'][0]:
        for rank, doc_id in enumerate(dense_results['ids'][0], start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (dense_weight / (k + rank))

    # Process Sparse Results
    if sparse_results and sparse_results['ids'][0]:
        for rank, doc_id in enumerate(sparse_results['ids'][0], start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (sparse_weight / (k + rank))

    # Sort documents by their accumulated RRF score in descending order
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs

# 6. Execute the Hybrid Search
query_text = "What does error ABC-123 mean?"

# Path A: Dense Semantic Search (Vector similarity match)
dense_res = collection.query(
    query_texts=[query_text],
    n_results=3
)

# Path B: Sparse/Keyword Search (Simulated using where-metadata string contains)
# Note: For production use cases, you can pre-tokenize or use a local BM25 index.
sparse_res = collection.query(
    query_texts=[query_text],
    where={"tags": {"$contains": "abc-123"}}, 
    n_results=3
)

# 7. Fuse results using RRF (Giving equal 1.0 weight here)
fused_rankings = reciprocal_rank_fusion(dense_res, sparse_res, k=60, dense_weight=1.0, sparse_weight=1.0)

# 8. Display Results
print(f"Query: '{query_text}'\n")
print("Fused Rankings (Doc ID, RRF Score):")
for doc_id, score in fused_rankings:
    # Retrieve original text snippet
    doc_text = collection.get(ids=[doc_id])['documents'][0]
    print(f"- {doc_id} (Score: {score:.5f}): \"{doc_text}\"")
