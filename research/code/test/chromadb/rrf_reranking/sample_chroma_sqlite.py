import sqlite3
import re
import chromadb
from chromadb.utils import embedding_functions
import ollama

# ---------------------------------------------------------
# 1. SETUP DATABASES & CONFIGURATION
# ---------------------------------------------------------
DB_NAME = "hybrid_search.db"
CHROMA_PATH = "./chroma_db"
OLLAMA_MODEL = "qwen3.5b-6-6:latest"

# Setup SQLite with FTS5 virtual table
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS multi_field_fts;")

# Creating an FTS5 table with the requested data fields.
# Non-searchable elements use 'UNINDEXED' to optimize the internal full-text registry.
cursor.execute("""
    CREATE VIRTUAL TABLE multi_field_fts USING fts5(
        uri UNINDEXED, 
        id UNINDEXED, 
        src UNINDEXED, 
        ts UNINDEXED, 
        type UNINDEXED, 
        latlon UNINDEXED, 
        loc UNINDEXED, 
        ppt UNINDEXED, 
        caption, 
        text
    );
""")
conn.commit()

# Setup ChromaDB for semantic vector embeddings
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="multi_field_collection", 
    embedding_function=default_ef
)

# ---------------------------------------------------------
# 2. SEEDING THE MULTI-FIELD DATA
# ---------------------------------------------------------
documents = [
    {
        "uri": "s3://bucket/doc1.txt", "id": "doc_001", "src": "web_scraper", "ts": "2026-09-28T10:00:00Z", 
        "type": "article", "latlon": "37.7749,-122.4194", "loc": "San Francisco", "ppt": "high_priority",
        "caption": "Decoupled Scaling Overview", 
        "text": "Microservices architecture allows decoupled scaling across complex server networks."
    },
    {
        "uri": "s3://bucket/doc2.txt", "id": "doc_002", "src": "internal_logs", "ts": "2026-09-28T11:30:00Z", 
        "type": "guide", "latlon": "40.7128,-74.0060", "loc": "New York", "ppt": "medium_priority",
        "caption": "AI Infrastructure", 
        "text": "The Python programming language is heavily utilized in data science and AI applications."
    },
    {
        "uri": "s3://bucket/doc3.txt", "id": "doc_003", "src": "web_scraper", "ts": "2026-09-28T12:15:00Z", 
        "type": "article", "latlon": "34.0522,-118.2437", "loc": "Los Angeles", "ppt": "low_priority",
        "caption": "Database Resiliency Insights", 
        "text": "Decoupled databases help maintain system stability when network load spikes suddenly."
    },
    {
        "uri": "s3://bucket/doc4.txt", "id": "doc_004", "src": "manual", "ts": "2026-09-28T14:00:00Z", 
        "type": "documentation", "latlon": "39.9042,116.4074", "loc": "Beijing", "ppt": "high_priority",
        "caption": "Local Intelligence Implementations", 
        "text": "The Qwen3.5 model is a highly efficient local LLM optimized for reasoning tasks."
    }
]

# Populate both structures
for doc in documents:
    # 1. Populate SQLite
    cursor.execute("""
        INSERT INTO multi_field_fts (uri, id, src, ts, type, latlon, loc, ppt, caption, text) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (doc["uri"], doc["id"], doc["src"], doc["ts"], doc["type"], doc["latlon"], doc["loc"], doc["ppt"], doc["caption"], doc["text"]))
    
    # 2. Populate ChromaDB
    # We combine 'caption' and 'text' as the core vector content, routing other components to metadata
    combined_text_payload = f"Caption: {doc['caption']}\nContent: {doc['text']}"
    metadata_payload = {
        "uri": doc["uri"], "src": doc["src"], "ts": doc["ts"], 
        "type": doc["type"], "latlon": doc["latlon"], "loc": doc["loc"], "ppt": doc["ppt"]
    }
    collection.add(
        ids=[doc["id"]], 
        documents=[combined_text_payload],
        metadatas=[metadata_payload]
    )

conn.commit()
print("Databases successfully seeded with custom schemas.")

# ---------------------------------------------------------
# 3. HYBRID SEARCH PIPELINE
# ---------------------------------------------------------
def search_sqlite_bm25(query_text, limit=3):
    """Searches indexed textual fields in SQLite via BM25."""
    clean_query = re.sub(r'[^\w\s]', '', query_text)
    # We query FTS5 specifically looking for structural hits inside caption OR text
    cursor.execute("""
        SELECT id FROM multi_field_fts 
        WHERE multi_field_fts MATCH ? 
        ORDER BY bm25(multi_field_fts) 
        LIMIT ?;
    """, (f"caption:{clean_query} OR text:{clean_query}", limit))
    return [row[0] for row in cursor.fetchall()]

def search_chroma_vector(query_text, limit=3):
    """Searches vectorized contexts inside ChromaDB."""
    results = collection.query(query_texts=[query_text], n_results=limit)
    return results['ids'][0] if results['ids'] else []

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    """Fuses results from keyword and vector structures."""
    rrf_scores = {}
    for rank, doc_id in enumerate(vector_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
    for rank, doc_id in enumerate(bm25_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs]

# ---------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------
query = "Tell me about decoupled scaling in microservices"
print(f"\nUser Query: '{query}'")

bm25_top_ids = search_sqlite_bm25(query)
vector_top_ids = search_chroma_vector(query)
fused_ids = reciprocal_rank_fusion(vector_top_ids, bm25_top_ids)

print(f"-> Vector Matches (IDs): {vector_top_ids}")
print(f"-> BM25 Matches (IDs):   {bm25_top_ids}")
print(f"-> Final RRF Ranking:    {fused_ids}")

# Build Context from the winning documents
context_chunks = []
for doc_id in fused_ids[:2]:  # Use top 2 matches
    cursor.execute("SELECT loc, caption, text FROM multi_field_fts WHERE id = ?;", (doc_id,))
    res = cursor.fetchone()
    if res:
        loc, caption, text = res
        context_chunks.append(f"[{caption} (Location: {loc})]: {text}")

context_string = "\n".join(context_chunks)

# Generation Prompt
prompt = f"""Use the following context snippets to answer the question accurately.
Context:
{context_string}

Question: {query}
Answer:"""

print("\n--- Interrogating Local Qwen3.5 Model ---")
try:
    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    print(response['response'])
except Exception as e:
    print(f"Ollama execution failed. Check your local service. Error: {e}")

conn.close()
