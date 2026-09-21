import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# 1. Initialize Mock Data and Models
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence and machine learning are transforming industries.",
    "Python is a popular programming language for data science and analytics.",
    "Advanced retrieval pipelines use hybrid search and reranking techniques.",
    "Vector databases like Chroma help store and query dense embeddings efficiently."
]

# Initialize Chroma (Dense Vector Store)
chroma_client = chromadb.Client()
emb_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.create_collection(name="hybrid_search_docs", embedding_function=emb_fn)

# Add documents to Chroma
collection.add(
    documents=documents,
    ids=[str(i) for i in range(len(documents))]
)

# Initialize BM25 (Sparse Text Search)
tokenized_corpus = [doc.lower().split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# Initialize Reranker (Cross-Encoder)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# 2. Define Pipeline Helpers
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """
    Combines two ranked lists using Reciprocal Rank Fusion (RRF).
    """
    rrf_scores = {}
    
    # Process dense ranks
    for rank, doc_id in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (k + (rank + 1)))
        
    # Process sparse ranks
    for rank, doc_id in enumerate(sparse_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (k + (rank + 1)))
        
    # Sort documents by descending RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs


# 3. Main Retrieval Pipeline
def advanced_retrieval_pipeline(original_query, query_variations):
    # Combine original query with variations to expand search scope
    all_queries = [original_query] + query_variations
    
    # Dictionaries to track best dense and sparse positions globally across variations
    dense_global_ranks = []
    sparse_global_ranks = []
    
    for q in all_queries:
        # --- A. Dense Retrieval (Chroma DB) ---
        dense_res = collection.query(query_texts=[q], n_results=3)
        dense_ids = dense_res['ids'][0] if dense_res['ids'] else []
        for doc_id in dense_ids:
            if doc_id not in dense_global_ranks:
                dense_global_ranks.append(doc_id)
                
        # --- B. Sparse Retrieval (BM25) ---
        tokenized_query = q.lower().split(" ")
        sparse_scores = bm25.get_scores(tokenized_query)
        # Get indices of top 3 scores
        top_sparse_indices = np.argsort(sparse_scores)[::-1][:3]
        for idx in top_sparse_indices:
            doc_id = str(idx)
            if doc_id not in sparse_global_ranks:
                sparse_global_ranks.append(doc_id)

    # --- C. Reciprocal Rank Fusion (RRF) ---
    rrf_ranked_docs = reciprocal_rank_fusion(dense_global_ranks, sparse_global_ranks)
    
    # Collect candidate text documents based on RRF top results
    candidate_ids = [doc_id for doc_id, score in rrf_ranked_docs]
    candidate_texts = [documents[int(doc_id)] for doc_id in candidate_ids]
    
    # --- D. Cross-Encoder Reranking ---
    # Construct pairs using the original user query for absolute relevance
    pairs = [[original_query, doc_text] for doc_text in candidate_texts]
    rerank_scores = reranker.predict(pairs)
    
    # Sort candidates by reranker output scores
    reranked_results = sorted(
        zip(candidate_ids, candidate_texts, rerank_scores), 
        key=lambda x: x[2], 
        reverse=True
    )
    
    return reranked_results


# 4. Execution Example
if __name__ == "__main__":
    # Original query and 3 user query variations
    user_query = "How to build advanced search pipelines?"
    variations = [
        "hybrid vector search with reranking",
        "chroma db dense and sparse retrieval code",
        "RRF fusion search architecture"
    ]
    
    print(f"--- Processing Pipeline for Query: '{user_query}' ---\n")
    final_results = advanced_retrieval_pipeline(user_query, variations)
    
    print("Final Top Reranked Results:")
    for rank, (doc_id, text, score) in enumerate(final_results, 1):
        print(f"{rank}. [ID: {doc_id}] [Score: {score:.4f}] -> {text}")
