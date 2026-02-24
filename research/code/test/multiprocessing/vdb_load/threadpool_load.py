import chromadb
from concurrent.futures import ThreadPoolExecutor

# Initialize Client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_collection")
l1 = list(range(1,5))
l2 =list(range(1,5))
# Sample data batches
data_batches = {'ids': l1, 'documents': l2}

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
