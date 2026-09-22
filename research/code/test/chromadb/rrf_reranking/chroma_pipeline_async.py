import json
import sqlite3
import asyncio
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import ollama

# 1. Configuration & Global Initializations
OLLAMA_MODEL = "qwen3.5b-6-6:latest"#"qwen2.5:7b"
DB_PATH = "async_scalable_store_4.db"
CHROMA_PATH = "./chroma_db_4"
BATCH_SIZE = 1000

# Initialize Persistent SQLite Database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL
)
""")

cursor.execute("""
CREATE VIRTUAL TABLE IF NOT EXISTS persistent_bm25_idx 
USING fts5(id UNINDEXED, text)
""")
conn.commit()

# Initialize Persistent Chroma DB Client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
emb_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="async_hybrid_search", embedding_function=emb_fn)

# Initialize Cross-Encoder Reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# 2. Batching & Stream Processing Implementation
def stream_and_index_documents(data_generator):
    """
    Consumes an item generator/stream, processes chunks in batches, 
    and writes to SQLite and Chroma.
    """
    sqlite_batch = []
    chroma_texts = []
    chroma_ids = []
    
    cursor.execute("SELECT COUNT(*) FROM documents")
    global_counter = cursor.fetchone()[0]
    
    print("⏳ Beginning streaming chunk insertion processes...")
    
    for text in data_generator:
        doc_id = str(global_counter)
        sqlite_batch.append((doc_id, text))
        chroma_texts.append(text)
        chroma_ids.append(doc_id)
        global_counter += 1
        
        if len(sqlite_batch) >= BATCH_SIZE:
            _flush_batch(sqlite_batch, chroma_texts, chroma_ids)
            sqlite_batch, chroma_texts, chroma_ids = [], [], []

    if sqlite_batch:
        _flush_batch(sqlite_batch, chroma_texts, chroma_ids)


def _flush_batch(sqlite_batch, chroma_texts, chroma_ids):
    cursor.executemany("INSERT OR IGNORE INTO documents (id, text) VALUES (?, ?)", sqlite_batch)
    cursor.executemany("INSERT OR IGNORE INTO persistent_bm25_idx (id, text) VALUES (?, ?)", sqlite_batch)
    conn.commit()
    collection.add(documents=chroma_texts, ids=chroma_ids)
    print(f"   Processed and committed stream batch chunk of size {len(sqlite_batch)}.")


# 3. Asynchronous LLM Query Expansion Engine
async def generate_query_variations_async(original_query: str) -> list[str]:
    """
    Asynchronously handles the Ollama network call using asyncio.to_thread
    to prevent blocking while generating query variations.
    """
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
        # Offload blocking Ollama network request to a separate worker thread
        response = await asyncio.to_thread(
            ollama.generate, model=OLLAMA_MODEL, prompt=prompt, options={"temperature": 0.3}
        )
        raw_text = response['response'].strip()
        
        if "```" in raw_text:
            raw_text = raw_text.split("```")
            if raw_text[1].startswith("json"):
                raw_text = raw_text[1][4:]
            else:
                raw_text = raw_text[1]
                
        variations = json.loads(raw_text.strip())
        return variations[:3]
    except Exception as e:
        print(f"⚠️ Falling back to default variations due to parsing error: {e}")
        return [f"{original_query} hybrid", f"{original_query} index", f"{original_query} vector"]


# 4. Synchronous Workers for Individual Query Execution
def run_dense_query(query: str) -> list[str]:
    """Synchronous worker that searches Chroma DB."""
    dense_res = collection.query(query_texts=[query], n_results=5)
    ids = []
    if dense_res and 'ids' in dense_res and dense_res['ids']:
        for nested_ids in dense_res['ids']:
            for doc_id in nested_ids:
                ids.append(doc_id)
    return ids


def run_sparse_query(query: str) -> list[str]:
    """Synchronous worker that searches SQLite FTS5."""
    ids = []
    clean_q = "".join([c if c.isalnum() or c.isspace() else " " for c in query]).strip()
    if clean_q:
        # Create a private connection per thread to ensure multi-threaded SQLite safety
        thread_conn = sqlite3.connect(DB_PATH)
        thread_cursor = thread_conn.cursor()
        try:
            thread_cursor.execute("""
                SELECT id FROM persistent_bm25_idx 
                WHERE persistent_bm25_idx MATCH ? 
                ORDER BY bm25(persistent_bm25_idx) ASC 
                LIMIT 5
            """, (clean_q,))
            ids = [str(row[0]) for row in thread_cursor.fetchall()]
        finally:
            thread_conn.close()
    return ids


# 5. Fixed Reciprocal Rank Fusion
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
    for rank, doc_id in enumerate(sparse_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
        
    return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)


# 6. Combined Asynchronous Pipeline Execution
async def advanced_retrieval_pipeline_async(original_query):
    # Step A: Query Expansion
    query_variations = await generate_query_variations_async(original_query)
    all_queries = [original_query] + query_variations
    print(f"   -> Executing Expanded Search Scope against: {all_queries}\n")
    
    # Step B: Concurrent Dense and Sparse Query Execution Tasks
    tasks = []
    for q in all_queries:
        # Offload both searches to the background thread pool concurrently
        tasks.append(asyncio.to_thread(run_dense_query, q))
        tasks.append(asyncio.to_thread(run_sparse_query, q))
    
    print("⚡ Executing all dense and sparse searches concurrently...")
    search_results = await asyncio.gather(*tasks)
    
    # Collate returned results into global rank tracking chains
    dense_global_ranks = []
    sparse_global_ranks = []
    
    # Tasks array interleaved: [Dense_Q0, Sparse_Q0, Dense_Q1, Sparse_Q1, ...]
    for idx, results in enumerate(search_results):
        if idx % 2 == 0:  # Dense results
            for doc_id in results:
                if doc_id not in dense_global_ranks:
                    dense_global_ranks.append(doc_id)
        else:  # Sparse results
            for doc_id in results:
                if doc_id not in sparse_global_ranks:
                    sparse_global_ranks.append(doc_id)

    # Step C: Merge Rankings using Reciprocal Rank Fusion
    rrf_ranked_docs = reciprocal_rank_fusion(dense_global_ranks, sparse_global_ranks)
    candidate_ids = [doc_id for doc_id, score in rrf_ranked_docs]
    
    if not candidate_ids:
        return []

    # Step D: Dynamic Text Fetching from SQL
    placeholders = ",".join("?" for _ in candidate_ids)
    cursor.execute(f"SELECT id, text FROM documents WHERE id IN ({placeholders})", candidate_ids)
    db_results = {str(row[0]): row[1] for row in cursor.fetchall()}
    
    candidate_texts = [db_results[doc_id] for doc_id in candidate_ids if doc_id in db_results]

    # Step E: Deep Cross-Encoder Reranking
    pairs = [[original_query, doc_text] for doc_text in candidate_texts]
    rerank_scores = await asyncio.to_thread(reranker.predict, pairs)
    
    reranked_results = sorted(
        zip(candidate_ids, candidate_texts, rerank_scores), 
        key=lambda x: x[2], 
        reverse=True
    )
    
    return reranked_results


# 7. Runtime Orchestration
async def main():
    def file_stream_simulator():
        large_mock_data = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence and machine learning are transforming industries.",
            "Python is a popular programming language for data science and analytics.",
            "Advanced retrieval pipelines use hybrid search and reranking techniques.",
            "Vector databases like Chroma help store and query dense embeddings efficiently."
        ]
        for text in large_mock_data:
            yield text

    # Seed data if empty
    cursor.execute("SELECT COUNT(*) FROM documents")
    if cursor.fetchone()[0] == 0:
        print("💾 Starting text streaming setup database pipelines...")
        stream_and_index_documents(file_stream_simulator())

    user_query = "How to build advanced search pipelines?"
    print(f"\n--- Running Asynchronous Disk Pipeline for: '{user_query}' ---\n")
    
    final_results = await advanced_retrieval_pipeline_async(user_query)
    
    print("\nFinal Top Reranked Results:")
    for rank, (doc_id, text, score) in enumerate(final_results, 1):
        print(f"{rank}. [ID: {doc_id}] [Rerank Score: {score:.4f}] -> {text}")
        
    conn.close()

if __name__ == "__main__":
    asyncio.run(main())
