import chromadb
import pandas as pd

# ... (assuming client and collection are initialized as above) ...
client = chromadb.PersistentClient("/mnt/zmdata/data/app-data/vectordb/")

collection = client.get_collection(name="multimodel_collection_images")


# Retrieve all data (be mindful of large collections)
all_data = collection.get(
    include=["metadatas", "embeddings", "documents"]
)

# Convert metadata to a pandas DataFrame
# This assumes your metadata has numerical fields you want to analyze
if all_data['metadatas']:
    df = pd.DataFrame(all_data['metadatas'])
    
    # Perform descriptive statistics on the DataFrame
    print("\nDescriptive statistics for metadata:")
    print(df.describe()) # .describe() works well on numeric columns

# You can also analyze the length of documents or other data points as needed
document_lengths = [len(doc) for doc in all_data['documents']]
df_lengths = pd.DataFrame(document_lengths, columns=['document_length'])
print("\nDescriptive statistics for document lengths:")
print(df_lengths.describe())
