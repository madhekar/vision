import json
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import ollama

# 1. Setup Local Configuration
OLLAMA_MODEL = "qwen3.5b-6-6:latest"#"qwen2.5:7b" 

# 2. Setup Mock Data
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence and machine learning are transforming industries.",
    "Python is a popular programming language for data science and analytics.",
    "Advanced retrieval pipelines use hybrid search and reranking techniques.",
    "Vector databases like Chroma help store and query dense embeddings efficiently."
]

chroma_client = chromadb.Client()
emb_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.create_collection(name="fixed_hybrid_search", embedding_function=emb_fn)

collection.add(
    documents=documents,
    ids=[str(i) for i in range(len(documents))]
)

tokenized_corpus = [doc.lower().split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# 3. Local LLM Variant Generator
def generate_query_variations_local(original_query: str) -> list[str]:
    prompt = f"""
    You are an AI assistant optimizing search retrieval queries.
    Given the user's original query, generate exactly 3 variations or phrasings 
    that cover different synonyms, technical terms, or perspectives.
    
    You must output your response strictly as a JSON array of strings. Do not include markdown formatting or extra text.
    Example output format: ["variation 1", "variation 2", "variation 3"]

    Original Query: {original_query}
    """
    
    print(f"🦙 Generating query variations with local model '{OLLAMA_MODEL}'...")
    try:
        response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, options={"temperature": 0.3})
        raw_text = response['response'].strip()
        
        # Strip markdown syntax if it leaks out
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
                
        variations = json.loads(raw_text.strip())
        return variations[:3]
    except Exception as e:
        print(f"⚠️ Falling back to default keywords list due to parse error: {e}")
        return [f"{original_query} hybrid", f"{original_query} chroma", f"{original_query} tutorial"]


# 4. FIXED Reciprocal Rank Fusion Function
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
    for rank, doc_id in enumerate(sparse_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
        
    # Key fix: item[1] forces sorting by score, not key name
    return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)


# 5. Combined Pipeline Execution Engine
def advanced_retrieval_pipeline(original_query):
    query_variations = generate_query_variations_local(original_query)
    all_queries = [original_query] + query_variations
    
    print(f"   -> Variations to check: {all_queries}\n")
    
    dense_global_ranks = []
    sparse_global_ranks = []
    
    for q in all_queries:
        # Dense Retrieval (Fixing nested list unpacking outputted from Chroma DB)
        dense_res = collection.query(query_texts=[q], n_results=3)
        if dense_res and 'ids' in dense_res and dense_res['ids']:
            # Chroma nested arrays structure: [['id1', 'id2']]
            for doc_id in dense_res['ids'][0]:
                if doc_id not in dense_global_ranks:
                    dense_global_ranks.append(doc_id)
                
        # Sparse Retrieval
        tokenized_query = q.lower().split(" ")
        sparse_scores = bm25.get_scores(tokenized_query)
        top_sparse_indices = np.argsort(sparse_scores)[::-1][:3]
        for idx in top_sparse_indices:
            doc_id = str(idx)
            if doc_id not in sparse_global_ranks:
                sparse_global_ranks.append(doc_id)

    # Calculate RRF Scores
    rrf_ranked_docs = reciprocal_rank_fusion(dense_global_ranks, sparse_global_ranks)
    
    candidate_ids = [doc_id for doc_id, score in rrf_ranked_docs]
    candidate_texts = [documents[int(doc_id)] for doc_id in candidate_ids]
    
    if not candidate_texts:
        return []

    # Final Cross-Encoder Reranking Execution Step
    pairs = [[original_query, doc_text] for doc_text in candidate_texts]
    rerank_scores = reranker.predict(pairs)
    
    # Sort correctly by score index position [2] descending
    reranked_results = sorted(
        zip(candidate_ids, candidate_texts, rerank_scores), 
        key=lambda x: x[2], 
        reverse=True
    )
    
    return reranked_results


# 6. Execute Application Demo
if __name__ == "__main__":
    user_query = "How to build advanced search pipelines?"
    
    print(f"--- Starting Fixed local Pipeline for: '{user_query}' ---\n")
    final_results = advanced_retrieval_pipeline(user_query)
    
    print("Final Top Reranked Results:")
    for rank, (doc_id, text, score) in enumerate(final_results, 1):
        print(f"{rank}. [ID: {doc_id}] [Rerank Score: {score:.4f}] -> {text}")
