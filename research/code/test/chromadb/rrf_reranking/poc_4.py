import chromadb
from chromadb import Schema, VectorIndexConfig, SparseVectorIndexConfig, K
from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction

# 1. Initialize your local or persistent instance
client = chromadb.PersistentClient(path="./chroma_db_4")

# 2. Define embedding models
# (Dense uses Chroma's default or a custom provider; Sparse uses BM25 here)
bm25_ef = ChromaBm25EmbeddingFunction()

# 3. Create a unified schema 
schema = (
    Schema()
    .create_index(
        VectorIndexConfig(space="cosine"), # Dense Semantic Index
    )
    .create_index(
        SparseVectorIndexConfig(
            source_key=K.DOCUMENT, 
            embedding_function=bm25_ef
        ), 
        "sparse_embedding" # Metadata key where sparse data lives
    )
)

# 4. Create the collection tied to this schema
collection = client.create_collection(
    name="hybrid_collection", 
    schema=schema
)

# add documents

collection.add(
    ids=["doc_1", "doc_2"],
    documents=[
        "Deep learning and neural networks power advanced AI models.",
        "To fix error SKU-404, reset the hardware routing firmware configuration."
    ]
)

# Performing Sparse + Dense Queries

from chromadb import Search, Knn, Rrf

# Define the query string
query_text = "AI neural networks error SKU-404"

# 1. Construct the Dense Semantic Rank
dense_rank = Knn(
    query=query_text, 
    return_rank=True
)

# 2. Construct the Sparse Keyword Rank (targeting your specific key)
sparse_rank = Knn(
    query=query_text, 
    key="sparse_embedding", 
    return_rank=True
)

# 3. Combine them using Reciprocal Rank Fusion (RRF)
hybrid_rank = Rrf(
    ranks=[dense_rank, sparse_rank],
    weights=[0.7, 0.3],  # Distribute importance: 70% semantic, 30% lexical
    k=60
)

# 4. Execute the combined search
search_pipeline = (
    Search()
    .rank(hybrid_rank)
    .limit(10)
    .select(K.DOCUMENT, K.SCORE)
)

results = collection.search(search_pipeline)
print(results)
