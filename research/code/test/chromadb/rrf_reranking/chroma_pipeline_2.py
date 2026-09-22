import json
import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import ollama

# 1. Configuration & Global Initializations
OLLAMA_MODEL = "qwen3.5b-6-6:latest"#qwen2.5:7b"
DB_PATH = "scalable_store_2.db"
CHROMA_PATH = "./chroma_db_2"
BATCH_SIZE = 1000  # Size of chunks used during streaming execution

# Initialize Persistent SQLite Database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create structured table for document metadata
cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL
)
""")

# Key Addition: Create a persistent BM25 (FTS5) full-text index table
# FTS5 uses a variant of Okapi BM25 inherently to calculate ranking scores natively on disk
cursor.execute("""
CREATE VIRTUAL TABLE IF NOT EXISTS persistent_bm25_idx 
USING fts5(id UNINDEXED, text)
""")
conn.commit()

# Initialize Persistent Chroma DB Client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
emb_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="production_hybrid_search", embedding_function=emb_fn)

# Initialize Cross-Encoder Reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# 2. Batching & Stream Processing Implementation
def stream_and_index_documents(data_generator):
    """
    Consumes an item generator/stream, processes chunks in batches, 
    and writes cleanly to SQLite and Chroma to handle millions of rows safely.
    """
    sqlite_batch = []
    chroma_texts = []
    chroma_ids = []
    
    # Track existing system offset to formulate global IDs
    cursor.execute("SELECT COUNT(*) FROM documents")
    global_counter = cursor.fetchone()[0]
    
    print("⏳ Beginning streaming chunk insertion processes...")
    
    for text in data_generator:
        doc_id = str(global_counter)
        sqlite_batch.append((doc_id, text))
        chroma_texts.append(text)
        chroma_ids.append(doc_id)
        global_counter += 1
        
        # When batch threshold is satisfied, dump to disk & clear heap allocation
        if len(sqlite_batch) >= BATCH_SIZE:
            _flush_batch(sqlite_batch, chroma_texts, chroma_ids)
            sqlite_batch, chroma_texts, chroma_ids = [], [], []

    # Flush final remaining records
    if sqlite_batch:
        _flush_batch(sqlite_batch, chroma_texts, chroma_ids)


def _flush_batch(sqlite_batch, chroma_texts, chroma_ids):
    """Internal helper to execute transactional batch insertions."""
    # Write to core relational table
    cursor.executemany("INSERT OR IGNORE INTO documents (id, text) VALUES (?, ?)", sqlite_batch)
    # Write to disk-backed persistent FTS5 BM25 search table
    cursor.executemany("INSERT OR IGNORE INTO persistent_bm25_idx (id, text) VALUES (?, ?)", sqlite_batch)
    conn.commit()
    
    # Write to vector index store
    collection.add(documents=chroma_texts, ids=chroma_ids)
    print(f"   Processed and committed stream batch chunk of size {len(sqlite_batch)}.")


# 3. Local LLM Query Expansion Engine
def generate_query_variations_local(original_query: str) -> list[str]:
    prompt = f"""
    You are an AI assistant optimizing search retrieval queries.
    Given the user's original query, generate exactly 3 variations or phrasings 
    that cover different synonyms, technical terms, or perspectives.
    
    You must output your response strictly as a JSON array of strings. Do not include markdown formatting or extra text.
    Example output format: ["variation 1", "variation 2", "variation 3"]

    Original Query: {original_query}
    """
    
    print(f"🦙 Generating query variations via '{OLLAMA_MODEL}'...")
    try:
        response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, options={"temperature": 0.3})
        raw_text = response['response'].strip()
        
        if "```" in raw_text:
            raw_text = raw_text.split("```")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
                
        variations = json.loads(raw_text.strip())
        return variations[:3]
    except Exception as e:
        print(f"⚠️ Falling back to default variations due to parsing error: {e}")
        return [f"{original_query} hybrid", f"{original_query} index", f"{original_query} vector"]


# 4. Fixed Reciprocal Rank Fusion
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
    for rank, doc_id in enumerate(sparse_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
        
    return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)


# 5. Production Retrieval Pipeline
def advanced_retrieval_pipeline(original_query):
    # Step A: Query Expansion
    query_variations = generate_query_variations_local(original_query)
    all_queries = [original_query] + query_variations
    print(f"   -> Executing Expanded Search Scope against: {all_queries}\n")
    
    dense_global_ranks = []
    sparse_global_ranks = []
    
    for q in all_queries:
        # 1. Dense Retrieval (Chroma DB)
        dense_res = collection.query(query_texts=[q], n_results=5)
        if dense_res and 'ids' in dense_res and dense_res['ids']:
            for nested_ids in dense_res['ids']:
                for doc_id in nested_ids:
                    if doc_id not in dense_global_ranks:
                        dense_global_ranks.append(doc_id)
                
        # 2. Native Sparse BM25 Retrieval (SQLite FTS5)
        # SQLite FTS5 uses a built-in BM25 algorithm exposed by ordering matches by 'bm25(table_name)' [1]
        # Token clean-up prevents syntax crashes from raw special characters
        clean_q = "".join([c if c.isalnum() or c.isspace() else " " for c in q]).strip()
        if clean_q:
            cursor.execute("""
                SELECT id FROM persistent_bm25_idx 
                WHERE persistent_bm25_idx MATCH ? 
                ORDER BY bm25(persistent_bm25_idx) ASC 
                LIMIT 5
            """, (clean_q,))
            
            for row in cursor.fetchall():
                doc_id = str(row[0])
                if doc_id not in sparse_global_ranks:
                    sparse_global_ranks.append(doc_id)

    # Step B: Merging Rankings using Reciprocal Rank Fusion
    rrf_ranked_docs = reciprocal_rank_fusion(dense_global_ranks, sparse_global_ranks)
    candidate_ids = [doc_id for doc_id, score in rrf_ranked_docs]
    
    if not candidate_ids:
        return []

    # Step C: Low-Memory Dynamic Text Fetching from SQL
    placeholders = ",".join("?" for _ in candidate_ids)
    cursor.execute(f"SELECT id, text FROM documents WHERE id IN ({placeholders})", candidate_ids)
    db_results = {str(row[0]): row[1] for row in cursor.fetchall()}
    
    # Maintain strict chronological array tracking matching the output sorted by RRF
    candidate_texts = [db_results[doc_id] for doc_id in candidate_ids if doc_id in db_results]

    # Step D: Deep Cross-Encoder Reranking
    pairs = [[original_query, doc_text] for doc_text in candidate_texts]
    rerank_scores = reranker.predict(pairs)
    
    reranked_results = sorted(
        zip(candidate_ids, candidate_texts, rerank_scores), 
        key=lambda x: x[2], 
        reverse=True
    )
    
    return reranked_results


# 6. Runtime Simulation Demo
if __name__ == "__main__":
    # Simulate a stream/generator of text data (e.g., streaming rows out of a raw data log)
    def file_stream_simulator():
        large_mock_data = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence and machine learning are transforming industries.",
            "Python is a popular programming language for data science and analytics.",
            "Advanced retrieval pipelines use hybrid search and reranking techniques.",
            "Vector databases like Chroma help store and query dense embeddings efficiently."
        ]
        # Yield lines continuously to simulate true stream data sources
        for text in large_mock_data:
            yield text

    # Seed data iteratively using streaming architecture if database index is empty
    cursor.execute("SELECT COUNT(*) FROM documents")
    if cursor.fetchone()[0] == 0:
        print("💾 Starting text streaming setup database pipelines...")
        stream_and_index_documents(file_stream_simulator())

    user_query = "How to build advanced search pipelines?"
    print(f"\n--- Running Persistent Disk Pipeline for: '{user_query}' ---\n")
    
    final_results = advanced_retrieval_pipeline(user_query)
    
    print("Final Top Reranked Results:")
    for rank, (doc_id, text, score) in enumerate(final_results, 1):
        print(f"{rank}. [ID: {doc_id}] [Rerank Score: {score:.4f}] -> {text}")
        
    conn.close()
