import sqlite3
import os
import chromadb
from chromadb.utils import embedding_functions
from collections import defaultdict
from chromadb.utils.data_loaders import ImageLoader

# --- CONFIGURATION ---
DB_PATH = "robust_search.db"
CHROMA_PATH = "./chroma_db_dense_chroma_sparce_sqllite"

if os.path.exists(DB_PATH): os.remove(DB_PATH)

# --- 1. INITIALIZE ENGINES ---
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("CREATE VIRTUAL TABLE sparse_index USING fts5(id UNINDEXED, content);")
conn.commit()

image_loader = ImageLoader()

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
default_ef = embedding_functions.DefaultEmbeddingFunction()
dense_collection = chroma_client.get_or_create_collection(
    name="dense_collection", 
    embedding_function=default_ef,
    metadata={"hnsw:space": "cosine", 
                  "hnsw:M" : 24, 
                  "hnsw:construction_ef": 200, 
                  "hnsw:search_ef": 100},
    data_loader=image_loader
)

# --- 2. BULK INSERTION ---
# {"uri": "/mnt/zmdata/home-media-app/data/input-data/img/ASSORT_K30/c47d67ff-6f83-5af2-ab2e-a48a62d8047b/IMG_7425.jpeg", 
# "id": "e3a10878-b740-4fc3-8d22-1646909712d0", 
# "src": "ASSORT_K30", 
# "ts": "1538460356.0", 
# "type": "people", 
# "latlon": "(32.968689, -117.184243)", 
# "loc": "Madhekar residence in Carmel Valley", 
# "ppt": "", 
# "caption": "", 
# "text": ""}

# {"uri": "/mnt/zmdata/home-media-app/data/input-data/img/ASSORT_K30/c47d67ff-6f83-5af2-ab2e-a48a62d8047b/IMG_7426.jpeg", 
# "id": "466b4b8b-a549-4006-82db-072713e4c9e0", 
# "src": "ASSORT_K30", 
# "ts": "1538460377.0", 
# "type": "people", 
# "latlon": 
# "(32.968689, -117.184243)", 
# "loc": "Madhekar residence in Carmel Valley",
# "ppt": "", 
# "caption": "", 
# "text": ""}
#
def insert_documents(batch_docs, batch_ids):
    cursor.executemany("INSERT INTO sparse_index (id, content) VALUES (?, ?);", zip(batch_ids, batch_docs))
    conn.commit()
    dense_collection.add(documents=batch_docs, ids=batch_ids)

# --- 3. RETRIEVAL & HYBRID FUSION (RRF) ---
def query_sparse(query_text, n_results=10):
    sql_query = """
        SELECT id, content FROM sparse_index 
        WHERE sparse_index MATCH ? 
        ORDER BY bm25(sparse_index) ASC LIMIT ?;
    """
    sanitized_query = " OR ".join(f'"{word}"' for word in query_text.split() if word.isalnum())
    if not sanitized_query: return []
    print("sanitized query: ", sanitized_query)
    cursor.execute(sql_query, (sanitized_query, n_results))
    return cursor.fetchall()

def query_dense(query_text, n_results=10):
    res = dense_collection.query(query_texts=[query_text], n_results=n_results)
    # Reformat to match sparse structure: [(id, content), ...]
    if not res['ids'] or not res['ids'][0]: return []
    return list(zip(res['ids'][0], res['documents'][0]))

def hybrid_search_rrf(query_text, k=60, top_n=3):
    """
    Executes dense and sparse searches, then combines them using Reciprocal Rank Fusion.
    k: Constant that penalizes low-ranked items (standard industry default is 60).
    top_n: Number of final fused documents to return.
    """
    # 1. Fetch deep candidate pools from both engines
    # We pull more candidates than 'top_n' to give the fusion algorithm enough crossover data
    candidate_pool_size = top_n * 3 
    
    sparse_results = query_sparse(query_text, n_results=candidate_pool_size)
    dense_results = query_dense(query_text, n_results=candidate_pool_size)
    
    # Trackers for RRF scores and document text mapping
    rrf_scores = defaultdict(float)
    doc_text_map = {}
    
    # 2. Score sparse candidates
    for rank, (doc_id, text) in enumerate(sparse_results, start=1):
        rrf_scores[doc_id] += 1.0 / (k + rank)
        doc_text_map[doc_id] = text
        
    # 3. Score dense candidates
    for rank, (doc_id, text) in enumerate(dense_results, start=1):
        rrf_scores[doc_id] += 1.0 / (k + rank)
        doc_text_map[doc_id] = text

    # 4. Sort candidates by their combined RRF score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 5. Format final output
    final_results = []
    for doc_id, score in sorted_docs[:top_n]:
        final_results.append({
            "id": doc_id,
            "text": doc_text_map[doc_id],
            "rrf_score": score
        })
        
    return final_results

# --- 4. EXECUTION SIMULATION ---

large_dataset_mock = [
 (
     'img_001',"The image captures a moment shared by four individuals. Kumar, with his glasses and red shirt, stands alongside Asha, who is wearing a white saree. Two women are also present in the picture; one of them can be seen holding a purse. They all appear to be posing for the photo with cheerful expressions on their faces. The setting seems serene, surrounded by nature, suggesting that they might be enjoying a day out or celebrating an occasion at this location."
 ),
 (
     'img_002',"In the image, there is a delightful beach scene featuring two young women standing on the sandy shore with the ocean visible behind them. The woman on the left, identified as Esha, is smiling broadly and appears to be enjoying her time at the beach, embodying a happy soul. Her companion is also smiling, indicating they are both having a pleasant experience.\n\nThey are positioned close together, suggesting a friendly relationship or camaraderie between them. The ocean in the background has calm waters with small waves, providing a serene and picturesque setting for their outing.\n\nThe reference to \"Madhekar residence in Carmel Valley\" might imply that this beach is located near or within the vicinity of such a residential area. However, it's important to note that the image does not provide direct evidence of the specific location being Madhekar's residence in Carmel Valley, and it could be merely an assumption based on the limited context provided."
 ),
  (
     'img_003',"The image features a young girl named Esha sitting at a table inside what appears to be the Madhekar Residence Home located in San Diego. She is wearing a blue top and seems to be in a joyful mood, as indicated by her bright smile. Her surroundings suggest an indoor setting with kitchen appliances visible in the background. The image conveys a sense of warmth and happiness associated with Esha's character, as well as the inviting ambiance of her home environment."
  ),
  (
     'img_004',"The image captures a delightful moment at the Madhekar residence in Carmel Valley. There are four individuals present - Esha, Anjali, another girl, and one more person whose name is not mentioned. They are all facing towards the camera, their smiles radiant as they look up. \n\nEsha, wearing a patterned top, stands to the left of the frame, her smile warm and inviting. In the center of the image, Anjali can be seen in a white shirt. She too is smiling broadly, adding to the cheerful atmosphere. To the right of the frame is another girl; she's also looking up at the camera with a big smile on her face.\n\nThe fourth person is located in the bottom left corner of the image. They are wearing glasses and are also smiling as they look towards the camera. The Madhekar residence, visible in the background, provides a homely backdrop to this cheerful gathering."
 ),
  (
     'img_005',"The image depicts a warm and intimate scene at the Madhekar residence in Carmel Valley. In the center of the frame, Esha, a happy soul, is seated on the floor with her legs stretched out in front of her. She's wearing a comfortable blue shirt that matches the lively energy she exudes.\n\nTo her right, another girl is sitting down as well, engaged in conversation or perhaps enjoying some quiet time together. The two girls seem to be having a pleasant interaction, contributing to the overall relaxed atmosphere of the scene.\n\nOn Esha's left, another person, Esha - the neutral soul, stands with an air of calmness and composure that contrasts with her twin's effervescent mood. She is dressed in a blue top as well, creating a sense of harmony within the image.\n\nThe kitchen counter in the background is cluttered with various items, including a cup and some food wrappers, indicating recent activity or ongoing preparation for a meal. The presence of these everyday objects adds a layer of authenticity to this snapshot of life at the Madhekar residence."
  )
]


""" large_dataset_mock = [
    ("doc_101", "Production architectures require database persistence and low latency configurations."),
    ("doc_102", "SQLite FTS5 provides memory efficient, disk-backed full-text search capabilities."),
    ("doc_103", "ChromaDB scales vector queries using high performance HNSW indexing structures."),
    ("doc_104", "Hybrid pipelines use reciprocal rank fusion to balance precision and semantic mapping.")
] """

ids, docs = zip(*large_dataset_mock)
insert_documents(list(docs), list(ids))

# This query has strong semantic matches for one doc, and exact keyword matches for another
search_query = "high performance full-text database index structures"
print(f"Executing Hybrid RRF Search for: '{search_query}'\n")

fused_results = hybrid_search_rrf(search_query, k=60, top_n=3)

for idx, doc in enumerate(fused_results, start=1):
    print(f"Rank {idx} | ID: {doc['id']} (RRF Score: {doc['rrf_score']:.5f})")
    print(f"Text: {doc['text']}\n")

conn.close()
