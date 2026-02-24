import chromadb
from concurrent.futures import ThreadPoolExecutor

# Initialize Client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_collection")

# Sample data batches
data_batches = [
    {"ids": ["id1"], "documents": ["doc1"]},
    {"ids": ["id2"], "documents": ["doc2"]},
    {"ids": ["id3"], "documents": ["doc3"]}
]

# Function to add to collection
def add_to_db(batch):
    collection.add(
        ids=batch["ids"],
        documents=batch["documents"]
    )
    print(f"Added {batch['ids']}")

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(add_to_db, data_batches)
