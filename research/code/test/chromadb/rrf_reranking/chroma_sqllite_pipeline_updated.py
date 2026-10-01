import re
import json
import time
import sqlite3
import asyncio
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import ollama

# 1. Configuration & Global Initializations
OLLAMA_MODEL = "qwen3.5b-6-6:latest" #"qwen2.5:7b"
DB_PATH = "/mnt/zmdata/home-media-app/data/app-data/sqllite/zm_image_idx.db"#"async_scalable_store.db"
CHROMA_PATH = "/mnt/zmdata/home-media-app/data/app-data/vectordb"#"./chroma_db"
BATCH_SIZE = 1000
TOP_QUERY_N = 3

# Initialize Persistent SQLite Database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# # 1. Create the main table to store all data fields
# cursor.execute(
#         """
#         CREATE TABLE IF NOT EXISTS documents (
#             uri TEXT,
#             id TEXT PRIMARY KEY,
#             src TEXT,
#             ts TEXT,
#             type TEXT,
#             latlon TEXT,
#             loc TEXT,
#             ppt TEXT,
#             caption TEXT,
#             text TEXT
#         )
#     """
#     )

# # 2. Create the FTS5 virtual table (externally content-backed for efficiency)
# # Since only 'text' is used for searching, it's the only indexed column.
# cursor.execute(
#         """
#         CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
#             text,
#             content='documents',
#             content_rowid='rowid'
#         )
#     """
#     )

# # Create triggers to keep the FTS index automatically updated on inserts
# cursor.execute(
#         """
#         CREATE TRIGGER IF NOT EXISTS t_documents_ai AFTER INSERT ON documents BEGIN
#             INSERT INTO documents_fts(rowid, text) VALUES (new.rowid, new.text);
#         END;
#     """
#     )
# conn.commit()

# Initialize Persistent Chroma DB Client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
#emb_fn =   #openclip embedding function!
embedding_function = OpenCLIPEmbeddingFunction()# embedding_functions.DefaultEmbeddingFunction()
#collection = chroma_client.get_or_create_collection(name="async_hybrid_search", embedding_function=emb_fn)
collection_images = chroma_client.get_or_create_collection(
      name="multimodal_collection_images", 
      embedding_function=embedding_function, 
      metadata={"hnsw:space": "cosine",
                "hnsw:M" : 24, 
                "hnsw:construction_ef": 200, 
                "hnsw:search_ef": 100},
      )
# Initialize Cross-Encoder Reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# 2. Production Batch Ingestion Engine (Ensures Persistence)
# def index_documents_batch(documents_list: list[str]):
#     """
#     Accepts a list of raw text strings, chunks them dynamically into 
#     configured BATCH_SIZE chunks, and executes atomic insertions into 
#     SQLite, SQLite FTS5 (BM25), and ChromaDB.
#     """
#     if not documents_list:
#         print("⚠️ Document input list is empty. Skipping execution.")
#         return

#     # Calculate global sequence tracking offset to safely append rows
#     cursor.execute("SELECT COUNT(*) FROM documents")
#     global_counter = cursor.fetchone()[0]
    
#     print(f"📦 Preparing to index {len(documents_list)} documents into persistent storage...")
    
#     for i in range(0, len(documents_list), BATCH_SIZE):
#         chunk = documents_list[i : i + BATCH_SIZE]
        
#         sqlite_batch = []
#         chroma_texts = []
#         chroma_ids = []
        
#         for text in chunk:
#             doc_id = str(global_counter)
#             sqlite_batch.append((doc_id, text))
#             chroma_texts.append(text)
#             chroma_ids.append(doc_id)
#             global_counter += 1

#         # Transactional SQLite writes
#         try:
#             cursor.execute("BEGIN TRANSACTION")
#             cursor.executemany("INSERT OR IGNORE INTO documents (id, text) VALUES (?, ?)", sqlite_batch)
#             cursor.executemany("INSERT OR IGNORE INTO persistent_bm25_idx (id, text) VALUES (?, ?)", sqlite_batch)
#             conn.commit()
#         except sqlite3.Error as e:
#             conn.rollback()
#             print(f"❌ SQLite database insertion rolled back due to error: {e}")
#             raise e
        
#         # Persistent Chroma write 
#         collection.add(documents=chroma_texts, ids=chroma_ids)
#         print(f"   ✅ successfully committed and vectorized batch chunk of size {len(chunk)}.")

# A basic list of common English stop words
STOP_WORDS = {"the", "is", "at", "which", "on", "and", "a", "an", "to", "in", "for", "with", "of"}

def clean_and_format_query(user_input: str) -> str:
    # 1. Lowercase and remove all non-alphanumeric/non-space characters
    clean_input = re.sub(r'[^\w\s]', '', user_input.lower())
    
    # 2. Tokenize and filter out common stop words
    words = [word for word in clean_input.split() if word not in STOP_WORDS]
    
    # 3. Format for FTS5 (joining words implies an 'AND' relationship)
    # Adding '*' turns them into prefix matches (e.g., "sql*" matches "sqlite")
    fts5_query = " ".join([f"{word}*" for word in words])
    
    return fts5_query

# 3. Asynchronous LLM Query Expansion Engine
# async def generate_query_variations_async(original_query: str) -> list[str]:
#     prompt = f"""
#     You are an AI assistant optimizing search retrieval queries.
#     Given the user's original query, generate exactly two variations or phrasings 
#     that cover different synonyms, technical terms, or perspectives.
    
#     You must output your response strictly as a JSON array of strings. Do not include markdown formatting or extra text.
#     Example output format: ["variation 1", "variation 2"]

#     Original Query: {original_query}
#     """
    
#     print(f"🦙 Generating query variations via '{OLLAMA_MODEL}'...")
#     try:
#         response = await asyncio.to_thread(
#             ollama.generate, model=OLLAMA_MODEL, prompt=prompt, options={"temperature": 0.3}
#         )
#         raw_text = response['response'].strip()
        
#         if "```" in raw_text:
#             parts = raw_text.split("```")
#             for part in parts:
#                 if part.strip().startswith("json"):
#                     raw_text = part.strip()[4:]
#                     break
#                 elif part.strip().startswith("["):
#                     raw_text = part.strip()
#                     break
                
#         variations = json.loads(raw_text.strip())
#         return variations[:2]
#     except Exception as e:
#         print(f"⚠️ Falling back to default variations due to parsing error: {e}")
#         return [f"{original_query} hybrid", f"{original_query} index", f"{original_query} vector"]


# 4. Thread-Safe Search Workers
def run_dense_query(query: str) -> list[str]:
    dense_res = collection_images.query(query_texts=[query], n_results=TOP_QUERY_N)
    print(f"---> dense query: {dense_res} \n")
    ids = []
    if dense_res and 'ids' in dense_res and dense_res['ids']:
        for nested_ids in dense_res['ids']:
            for doc_id in nested_ids:
                ids.append(doc_id)
    return ids


def run_sparse_query(query: str) -> list[str]:
    ids = []
    #clean_q = "".join([c if c.isalnum() or c.isspace() else " " for c in query]).strip()
    clean_q = clean_and_format_query(query).strip()
    if clean_q:
        thread_conn = sqlite3.connect(DB_PATH)
        #thread_conn.execute("PRAGMA journal_mode=WAL;")
        thread_cursor = thread_conn.cursor()
        try:
            thread_cursor.execute("""
        SELECT d.id, d.uri, d.caption, d.text, d.ts, bm25(documents_fts) as rank
        FROM documents_fts df
        JOIN documents d ON df.rowid = d.rowid
        WHERE documents_fts MATCH ?
        ORDER BY rank ASC
        LIMIT ?;
    """, (clean_q, TOP_QUERY_N))
            ids = [str(row[0]) for row in thread_cursor.fetchall()]
            print(f"--->sparse query: {ids} clean-q: {clean_q} \n")
        finally:
            thread_conn.close()
    return ids


# 5. Reciprocal Rank Fusion
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    print(f"RRF: dense: {dense_results} sparse: {sparse_results}")
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
    for rank, doc_id in enumerate(sparse_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
        
    return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)


# 6. Combined Asynchronous Pipeline Execution
async def advanced_retrieval_pipeline_async(original_query):
    #query_variations = await generate_query_variations_async(original_query)
    all_queries = [original_query] #+ query_variations
    print(f"   -> Executing Expanded Search Scope against: {all_queries}\n")
    
    tasks = []
    for q in all_queries:
        tasks.append(asyncio.to_thread(run_dense_query, q))
        tasks.append(asyncio.to_thread(run_sparse_query, q))
    
    print("⚡ Executing all dense and sparse searches concurrently...")
    search_results = await asyncio.gather(*tasks)
    
    dense_global_ranks = []
    sparse_global_ranks = []
    
    for idx, results in enumerate(search_results):
        if idx % 2 == 0:
            for doc_id in results:
                if doc_id not in dense_global_ranks:
                    dense_global_ranks.append(doc_id)
        else:
            for doc_id in results:
                if doc_id not in sparse_global_ranks:
                    sparse_global_ranks.append(doc_id)

    rrf_ranked_docs = reciprocal_rank_fusion(dense_global_ranks, sparse_global_ranks)
    candidate_ids = [doc_id for doc_id, score in rrf_ranked_docs]
    
    if not candidate_ids:
        return []

    # Dynamic low-heap text retrieval from SQLite
    placeholders = ",".join("?" for _ in candidate_ids)
    cursor.execute(f"SELECT id, text, uri, caption, ts, latlon, loc FROM documents WHERE id IN ({placeholders})", candidate_ids)
    # list_iter = []
    # columns = [col[0] for col in cursor.description]
    # print(f"---columns: {columns}")
    # results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    # print(f"---results: {results}")
    # db_results = [{d['id']:d} for d in results]

    #print(f"---> {exp_result}")
    # for item in cursor.fetchall():
    #     print(f"***item: {item}")
    #     list_iter.append({k: item[k] for k in item})
 
    #print([row for row in cursor.fetchall()])
    db_results = {str(row[0]): row[1] for row in cursor.fetchall()}
    
    candidate_texts = [db_results[doc_id] for doc_id in candidate_ids if doc_id in db_results]

    # Deep Cross-Encoder Reranking
    pairs = [[original_query, doc_text] for doc_text in candidate_texts]
    rerank_scores = await asyncio.to_thread(reranker.predict, pairs)
    
    reranked_results = sorted(
        zip(candidate_ids, candidate_texts, rerank_scores), 
        key=lambda x: x[2], 
        reverse=True
    )
    
    return reranked_results


# 7. Orchestrated Runtime Execution Loop
async def main():
    start_time = time.perf_counter()
 
    # Populate index if database files don't have records yet
    #cursor.execute("SELECT * FROM documents")

    user_query = "Esha and Shibangi"#"Working on the Apple mac while eating an Apple." #"Esha dressed in traditional Indian attire." #"How to build advanced search pipelines?"
    print(f"\n--- Running Asynchronous Disk Pipeline for: '{user_query}' ---\n")
    
    final_results = await advanced_retrieval_pipeline_async(user_query)

    print("\n***Reranked Results***\n")
    for rank, (doc_id, text, score) in enumerate(final_results, 1):
        print(f"{rank}. [ID: {doc_id}] [Rerank Score: {score:.4f}] -> {text}\n")
        
    conn.close()

    end_time = time.perf_counter()

    print(f"Elapsed time: {(end_time -  start_time):.6f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
