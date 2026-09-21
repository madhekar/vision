import json
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import ollama

# 1. Setup Local Configurations
# Using Qwen 2.5 (change to 'qwen2.5:1.5b' or 'llama3' depending on your hardware)
OLLAMA_MODEL = "qwen2.5:7b" 

# 2. Setup Mock Data and Pipeline Elements
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence and machine learning are transforming industries.",
    "Python is a popular programming language for data science and analytics.",
    "Advanced retrieval pipelines use hybrid search and reranking techniques.",
    "Vector databases like Chroma help store and query dense embeddings efficiently."
]

# Chroma setup
chroma_client = chromadb.Client()
emb_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.create_collection(name="local_hybrid_search", embedding_function=emb_fn)

collection.add(
    documents=documents,
    ids=[str(i) for i in range(len(documents))]
)

# BM25 & Reranker setup
tokenized_corpus = [doc.lower().split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# 3. Local LLM Query Generation using Ollama
def generate_query_variations_local(original_query: str) -> list[str]:
    """
    Uses a local Ollama model to generate exactly 3 distinct search variations.
    Instructs the model to output valid JSON for deterministic parsing.
    """
    prompt = f"""
    You are an AI assistant optimizing search retrieval queries.
    Given the user's original query, generate exactly 3 variations or phrasings 
    that cover different synonyms, technical terms, or perspectives.
    
    You must output your response strictly as a JSON array of strings. Do not include markdown formatting or extra text.
    Example output format: ["variation 1", "variation 2", "variation 3"]

    Original Query: {original_query}
    """
    
    print(f"🦙 Generating query variations with local model '{OLLAMA_MODEL}'...")
    
    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={
            "temperature": 0.3 # Keep creative variation low for relevant search targets
        }
    )
    
    raw_text = response['response'].strip()
    
    # Clean markdown code blocks if the local model adds them anyway
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    
    try:
        variations = json.loads(raw_text.strip())
        return variations[:3]
    except Exception as e:
        print(f"⚠️ Error parsing JSON from local LLM output. Falling back to basic phrasings. Error: {e}")
        # Fallback list if the local model fails structural formatting constraints
        return [f"{original_query} keywords", f"{original_query} tutorial", f"explain {original_query}"]


# 4. Pipeline Fusion Helper
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (k + (rank + 1)))
    for rank, doc_id in enumerate(sparse_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (k + (rank + 1)))
    return sorted(rrf_scores.items(), key=lambda item: item, reverse=True)


# 5. Combined Local Retrieval Pipeline
def advanced_retrieval_pipeline(original_query):
    # Step A: Local Query expansion
    query_variations = generate_query_variations_local(original_query)
    all_queries = [original_query] + query_variations
    
    print(f"   -> Local Variations Generated: {query_variations}\n")
    
    dense_global_ranks = []
    sparse_global_ranks = []
    
    # Step B: Multi-query Search execution
    for q in all_queries:
        # Dense Retrieval (Chroma DB)
        dense_res = collection.query(query_texts=[q], n_results=3)
        dense_ids = dense_res['ids'] if dense_res['ids'] else []
        for doc_id in dense_ids:
            if doc_id not in dense_global_ranks:
                dense_global_ranks.append(doc_id)
                
        # Sparse Retrieval (BM25)
        tokenized_query = q.lower().split(" ")
        sparse_scores = bm25.get_scores(tokenized_query)
        top_sparse_indices = np.argsort(sparse_scores)[::-1][:3]
        for idx in top_sparse_indices:
            doc_id = str(idx)
            if doc_id not in sparse_global_ranks:
                sparse_global_ranks.append(doc_id)

    # Step C: Merging using RRF
    rrf_ranked_docs = reciprocal_rank_fusion(dense_global_ranks, sparse_global_ranks)
    candidate_ids = [doc_id for doc_id, score in rrf_ranked_docs]
    candidate_texts = [documents[int(doc_id)] for doc_id in candidate_ids]
    
    if not candidate_texts:
        return []

    # Step D: Cross-Encoder Reranking
    pairs = [[original_query, doc_text] for doc_text in candidate_texts]
    rerank_scores = reranker.predict(pairs)
    
    reranked_results = sorted(
        zip(candidate_ids, candidate_texts, rerank_scores), 
        key=lambda x: x, 
        reverse=True
    )
    
    return reranked_results


# 6. Run the local Pipeline
if __name__ == "__main__":
    user_query = "How to build advanced search pipelines?"
    
    print(f"--- Starting Local Pipeline for Query: '{user_query}' ---\n")
    final_results = advanced_retrieval_pipeline(user_query)
    
    print("Final Top Reranked Results:")
    for rank, (doc_id, text, score) in enumerate(final_results, 1):
        print(f"{rank}. [ID: {doc_id}] [Score: {score:.4f}] -> {text}")
