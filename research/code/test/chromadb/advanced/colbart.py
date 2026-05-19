import chromadb
from ragatouille import RAGPretrainedModel

# 1. Setup ChromaDB client and collection
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="knowledge_base")

# Add some documents to Chroma
documents = [
    "ColBERT uses late interaction to measure text similarity at the token level.",
    "ChromaDB is an open-source embedding database for retrieval-augmented generation.",
    "Late interaction models compute MaxSim scores between query tokens and document tokens.",
    "RAG systems combine vector search with large language models."
]
collection.add(
    documents=documents,
    ids=["doc1", "doc2", "doc3", "doc4"]
)

# 2. Perform initial retrieval using Chroma (e.g., getting top 3 candidates)
query = "What is late interaction in ColBERT?"

# In real scenarios, use your embedding model here. For this example, 
# we query directly with strings (Chroma's default will handle if configured)
results = collection.query(
    query_texts=[query],
    n_results=3
)
retrieved_docs = results["documents"][0]

# 3. Rerank the retrieved results using ColBERT (via RAGatouille)
# This loads a pre-trained ColBERT model
reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Rerank the subset of documents from Chroma
reranked_results = reranker.rerank(
    query=query,
    documents=retrieved_docs
)

# 4. Display the results
print("Query:", query)
for rank, item in enumerate(reranked_results, 1):
    print(f"\nRank {rank}: Score {item['score']:.4f}")
    print(f"Document: {item['content']}")
