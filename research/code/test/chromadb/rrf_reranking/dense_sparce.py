import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import spacy

# 1. Initialize NLP tokenizer for BM25
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def tokenize(text):
    return [token.lemma_.lower() for token in nlp(text) if not token.is_space and not token.is_punct]

# Sample data to index
documents = [
    "Vector databases are efficient for semantic search.",
    "BM25 is a keyword-based retrieval algorithm based on TF-IDF.",
    "Hybrid search combines dense and sparse retrieval methods for better accuracy.",
    "ChromaDB is an open-source embedding database."
]
doc_ids = [f"doc_{i}" for i in range(len(documents))]

# 2. Setup Dense Collection (ChromaDB)
chroma_client = chromadb.Client()
# Using Chroma's default embedding function (all-MiniLM-L6-v2)
default_ef = embedding_functions.DefaultEmbeddingFunction()

dense_collection = chroma_client.create_collection(
    name="dense_collection", 
    embedding_function=default_ef
)

# Add documents to dense collection
dense_collection.add(
    documents=documents,
    ids=doc_ids
)

# 3. Setup Sparse Index (BM25)
tokenized_corpus = [tokenize(doc) for doc in documents]
bm25_index = BM25Okapi(tokenized_corpus)


# 4. Querying Functions
def query_dense(query_text, n_results=2):
    results = dense_collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results

def query_sparse(query_text, n_results=2):
    tokenized_query = tokenize(query_text)
    # Get similarity scores for all documents
    scores = bm25_index.get_scores(tokenized_query)
    
    # Pair documents and IDs with their scores, then sort
    scored_docs = sorted(
        zip(doc_ids, documents, scores), 
        key=lambda x: x[2], 
        reverse=True
    )
    return scored_docs[:n_results]


# --- Example Execution ---
query = "What is hybrid search?"

print(f"Query: '{query}'\n")

print("--- Dense Retrieval Results ---")
dense_res = query_dense(query)
for doc, doc_id in zip(dense_res['documents'][0], dense_res['ids'][0]):
    print(f"[{doc_id}]: {doc}")

print("\n--- Sparse (BM25) Retrieval Results ---")
sparse_res = query_sparse(query)
for doc_id, doc, score in sparse_res:
    print(f"[{doc_id}] (Score: {score:.4f}): {doc}")
