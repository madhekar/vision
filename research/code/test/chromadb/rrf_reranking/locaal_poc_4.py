import chromadb
from rank_bm25 import BM25Okapi  # pip install rank-bm25

# 1. Initialize normal local Chroma client for dense embeddings
client = chromadb.PersistentClient(path="./chroma_local")
collection = client.get_or_create_collection(name="local_hybrid_search")

# Sample corpus
documents = [
    "Deep learning and neural networks power advanced AI models.",
    "To fix error SKU-404, reset the hardware routing firmware configuration."
]
doc_ids = ["doc_1", "doc_2"]

# Add documents to Chroma (for Dense semantic search)
collection.add(ids=doc_ids, documents=documents)

# 2. Initialize BM25 locally for sparse keyword search
tokenized_corpus = [doc.lower().split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# 3. Client-side Reciprocal Rank Fusion (RRF) algorithm
def reciprocal_rank_fusion(dense_results, sparse_results, k=60, dense_weight=0.7, sparse_weight=0.3):
    fused_scores = {}
    
    # Process dense ranks
    for rank, doc_id in enumerate(dense_results):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + dense_weight * (1.0 / (k + rank + 1))
        
    # Process sparse ranks
    for rank, doc_id in enumerate(sparse_results):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + sparse_weight * (1.0 / (k + rank + 1))
        
    # Sort documents by descending fused score
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

# 4. Perform the Hybrid Query
query_text = "AI neural networks error SKU-404"

# Path A: Local Dense Query
dense_res = collection.query(query_texts=[query_text], n_results=10)
dense_ranked_ids = dense_res['ids'][0] if dense_res['ids'] else []

# Path B: Local Sparse Query
tokenized_query = query_text.lower().split(" ")
sparse_scores = bm25.get_scores(tokenized_query)
# Pair IDs with scores and sort them to get the ranks
sparse_ranked = sorted(zip(doc_ids, sparse_scores), key=lambda x: x[1], reverse=True)
sparse_ranked_ids = [doc_id for doc_id, score in sparse_ranked if score > 0]

# Path C: Fuse the results together
final_rankings = reciprocal_rank_fusion(dense_ranked_ids, sparse_ranked_ids)
print("Fused Results (ID, Score):", final_rankings)
