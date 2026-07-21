import chromadb
import json
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

class ChromaQuerier:
    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        """Initialize the ChromaDB client and load the collection."""
        self.client = chromadb.PersistentClient(path=persist_directory)
            # openclip embedding function!
        self.embedding_function = OpenCLIPEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(name=collection_name, 
                                                               embedding_function=self.embedding_function)

    def get_collection_count(self) -> int:
        """Return the total number of documents in the collection."""
        return self.collection.count()

    def query_collection(self, query_texts: list, n_results: int = 2) -> dict:
        """Return semantic similarity search results for given query texts."""
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results
        )

    def query_with_metadata(self, query_texts: list, where_filter: dict, n_results: int = 2) -> dict:
        """
        Return similarity search results with complex metadata filtering.
        
        Example where_filter:
        {"$and": [{"category": {"$eq": "research"}}, {"year": {"$gte": 2024}}]}
        """
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where_filter
        )

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":

    chroma_path = "/mnt/zmdata/home-media-app/data/app-data/vectordb/"
    img_collection_name = "multimodal_collection_images"

    querier = ChromaQuerier(collection_name=img_collection_name, persist_directory=chroma_path)
    
    # 1. Get count
    print(f"\n Total documents: \n {querier.get_collection_count()}")
    
    # 2. Basic Query
    results = querier.query_collection(query_texts=["esha"], n_results=2)
    
    print("\n Basic Query Results: \n", json.dumps(results, indent=2))
    
    # 3. Complex Metadata Query (e.g., category is 'research' AND year >= 2024)
    metadata_filter = {
        "$and": [
            {"src": {"$eq": "ASSORT_K30"}},
            {"ts": {"$gte": 946717260}}
        ]
    }
    filtered_results = querier.query_with_metadata(
        query_texts=["neural networks berkeley"], 
        where_filter=metadata_filter,
        n_results=5
    )
    print("\n Filtered Metadata Results: \n", json.dumps(filtered_results, indent=2))
