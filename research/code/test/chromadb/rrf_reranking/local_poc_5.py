import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sklearn.feature_extraction.text import HashingVectorizer

# 1. Initialize your single local instance
client = chromadb.PersistentClient(path="./chroma_local_store_5")

# 2. Define the exact Sparse Vector Layout using a Hashing Vectorizer
# This avoids storing or loading an external vocabulary mapping dictionary.
# We set 4096 dimensions, which is large enough to prevent hash collisions 
# while keeping query speeds lightning fast.
hash_vectorizer = HashingVectorizer(n_features=4096, alternate_sign=False, norm=None)


# 3. Create the Custom Dual Embedding Function
class LocalHybridEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, mode="dense"):
        self.mode = mode
        # Initialize default dense model natively supported by Chroma
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self.dense_ef = DefaultEmbeddingFunction()

    def __call__(self, input: Documents) -> Embeddings:
        if self.mode == "dense":
            # Native dense embedding math
            return self.dense_ef(input)
        elif self.mode == "sparse":
            # Proper sparse mapping to a fixed high-dimensional lexical space
            sparse_matrix = hash_vectorizer.transform(input)
            return sparse_matrix.toarray().tolist()


# 4. Bind the correct functions to individual structural collections
dense_collection = client.get_or_create_collection(
    name="dense_semantic_store",
    embedding_function=LocalHybridEmbeddingFunction(mode="dense")
)

sparse_collection = client.get_or_create_collection(
    name="sparse_lexical_store",
    embedding_function=LocalHybridEmbeddingFunction(mode="sparse")
)


# 5. Adding Document Data safely across the Single Instance
doc_ids = ["doc_1", "doc_2"]
documents_list = [
    "Deep learning and neural networks power advanced AI models.",
    "To fix error SKU-404, reset the hardware routing firmware configuration."
]

# When adding text, Chroma executes our custom wrapper logic automatically
dense_collection.add(ids=doc_ids, documents=documents_list)
sparse_collection.add(ids=doc_ids, documents=documents_list)


# --- HIGHLY PERFORMANT LOCAL HYBRID RETRIEVAL ---
def perform_hybrid_query(query_text: str, top_n: int = 10, k: int = 60):
    # Route 1: Rapid Semantic Dense Query
    dense_res = dense_collection.query(query_texts=[query_text], n_results=top_n)
    dense_ids = dense_res['ids'][0] if dense_res['ids'] else []

    # Route 2: Pure Key-Word Sparse Vector Match
    sparse_res = sparse_collection.query(query_texts=[query_text], n_results=top_n)
    sparse_ids = sparse_res['ids'][0] if sparse_res['ids'] else []

    # Route 3: Reciprocal Rank Fusion Execution
    fused_scores = {}
    for rank, doc_id in enumerate(dense_ids):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 0.7 * (1.0 / (k + rank + 1))
    
    for rank, doc_id in enumerate(sparse_ids):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 0.3 * (1.0 / (k + rank + 1))
        
    # Sort results by the unified rank output
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


# Execute the pipeline
results = perform_hybrid_query("AI neural networks error SKU-404")
print("Proper Single-Instance Hybrid Results:", results)
