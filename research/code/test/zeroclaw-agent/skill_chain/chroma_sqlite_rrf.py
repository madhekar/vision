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


def init_global(chroma_path, sqllite_path):
    # 1. Configuration & Global Initializations

    top_n_results = 9


    # Initialize Persistent SQLite Database
    conn = sqlite3.connect(sqllite_path)
    sqlite_cursor = conn.cursor()

    # Initialize Persistent Chroma DB Client
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    #emb_fn =   #openclip embedding function!
    embedding_function = OpenCLIPEmbeddingFunction()

    collection_images = chroma_client.get_or_create_collection(
        name="multimodal_collection_images", 
        embedding_function=embedding_function, 
        metadata={"hnsw:space": "cosine",
                    "hnsw:M" : 24, 
                    "hnsw:construction_ef": 200, 
                    "hnsw:search_ef": 100},
        )
    # Initialize Cross-Encoder Reranker
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") #cross-encoder/ms-marco-MiniLM-L-6-v2
    return (conn, sqlite_cursor, collection_images, reranker, top_n_results)

def clean_and_format_query(user_input: str) -> str:
    # A basic list of common English stop words
    STOP_WORDS = {"the", "is", "at", "which", "on", "and", "a", "an", "to", "in", "for", "with", "of"}
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
def run_dense_query(collection_images, query: str, top_n_results) -> list[str]:
    dense_res = collection_images.query(query_texts=[query], n_results=top_n_results)
    #print(f"---> dense query: {dense_res} \n")
    ids = []
    if dense_res and 'ids' in dense_res and dense_res['ids']:
        for nested_ids in dense_res['ids']:
            for doc_id in nested_ids:
                ids.append(doc_id)
    return ids


def run_sparse_query(sqlite_path, query: str, top_n_results) -> list[str]:
    ids = []
    #clean_q = "".join([c if c.isalnum() or c.isspace() else " " for c in query]).strip()
    clean_q = clean_and_format_query(query).strip()
    if clean_q:
        thread_conn = sqlite3.connect(sqlite_path)
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
    """, (clean_q, top_n_results))
            ids = [str(row[0]) for row in thread_cursor.fetchall()]
            #print(f"--->sparse query: {ids} clean-q: {clean_q} \n")
        finally:
            thread_conn.close()
    return ids


# 5. Reciprocal Rank Fusion
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    #print(f"RRF: dense: {dense_results} sparse: {sparse_results}")
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
    for rank, doc_id in enumerate(sparse_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
        
    return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)


# 6. Combined Asynchronous Pipeline Execution
async def advanced_retrieval_pipeline_async(original_query, collection_images, sqlite_path, cursor, reranker, top_n_results):
    #query_variations = await generate_query_variations_async(original_query)
    all_queries = [original_query] #+ query_variations
    #print(f"   -> Executing Expanded Search Scope against: {all_queries}\n")
    
    tasks = []
    for q in all_queries:
        tasks.append(asyncio.to_thread(run_dense_query, collection_images, q, top_n_results))
        tasks.append(asyncio.to_thread(run_sparse_query, sqlite_path, q, top_n_results))
    
    #print("⚡ Executing all dense and sparse searches concurrently...")
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
    #cursor.execute(f"SELECT id, text FROM documents WHERE id IN ({placeholders})", candidate_ids)
    cursor.execute(f"SELECT id, text, uri, caption, ts, latlon, loc, ppt FROM documents WHERE id IN ({placeholders})", candidate_ids)

    columns = [col[0] for col in cursor.description]

    rows =  [row for row in cursor.fetchall()]
    results = [dict(zip(columns, row)) for row in rows]

    lookup_results = [{d['id']:d for d in results}]
    dr = [{k:v  for k, v in d.items()} for d in lookup_results]
    
    db_results = {str(row[0]): row[1] for row in rows}
    #print(f"db_results---> {db_results}")
    candidate_texts = [db_results[doc_id] for doc_id in candidate_ids if doc_id in db_results]

    # Deep Cross-Encoder Reranking
    pairs = [[original_query, doc_text] for doc_text in candidate_texts]
    rerank_scores = await asyncio.to_thread(reranker.predict, pairs)

    items = list(map(dr[0].get, candidate_ids))

    reranked_results = sorted(
        zip(rerank_scores, items), 
        key=lambda x: x[0], 
        reverse=True
    )
    return reranked_results


# 7. Orchestrated Runtime Execution Loop
async def rrf_query(user_query, sqlite_path, sqlite_cursor, collection_images, reranker, rmax):
    start_time = time.perf_counter()

    #(conn, sqlite_cursor, collection_images, reranker, top_n_results) = init_global(chroma_path, sqlite_path)
 
    final_results = await advanced_retrieval_pipeline_async(user_query, collection_images, sqlite_path, sqlite_cursor, reranker, rmax)

    #print("\n***Reranked Results***\n")
    items = []
    for rank, (score, item) in enumerate(final_results, 1):
        items.append(item)
        #print(f"{rank}. [Rerank Score: {score:.4f}] item-> {item}\n")
    sqlite_cursor.close()

    end_time = time.perf_counter()

    #print(f"Elapsed time: {(end_time -  start_time):.6f} seconds")

    return items

# cu.rerank_image_text_search(rr_model, modalityTxt, cImgs, rmax=50, top_k=30)
def rerank_rrf_image_text_search(reranker, user_query, collection_images, sqlite_path, rmax=50, top_k=5):   
    conn = sqlite3.connect(sqlite_path)
    #thread_conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    #print(f"\n--- Running Asynchronous Disk Pipeline for: '{user_query}' ---\n")
    result = asyncio.run(rrf_query(user_query, sqlite_path, cursor, collection_images, reranker,  rmax))    
    return result[:top_k]

if __name__ == "__main__":    
    OLLAMA_MODEL = "qwen3.5b-6-6:latest" #"qwen2.5:7b"
    DB_PATH = "/mnt/zmdata/home-media-app/data/app-data/sqlite/zm_image_idx.db"
    CHROMA_PATH = "/mnt/zmdata/home-media-app/data/app-data/vectordb"
    (conn, sqlite_cursor, collection_images, reranker, top_n_results) =init_global(CHROMA_PATH, DB_PATH)
    user_query = "Esha and Shibangi"#"Working on the Apple mac while eating an Apple." r
    #"Esha dressed in traditional Indian attire." 
    # #"How to build advanced search pipelines?"
    rlist = rerank_rrf_image_text_search(reranker, user_query, collection_images, DB_PATH, sqlite_cursor, rmax=50, top_k=10)
    for i, rl in enumerate(rlist):
        print(f"{i}->{rl}\n")
