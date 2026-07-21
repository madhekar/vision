import chromadb
from chromadb.config import Settings

class ChromaQuerier:
    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        """Initialize the ChromaDB client and load the collection."""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def get_collection_count(self) -> int:
        """Return the total number of documents in the collection."""
        return self.collection.count()

    def query_collection(self, query_texts: list, n_results: int = 5) -> dict:
        """Return semantic similarity search results for given query texts."""
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results
        )

    def query_with_metadata(self, query_texts: list, where_filter: dict, n_results: int = 5) -> dict:
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
    chroma_path = "/mnt/zmedia/home-media-app/data/app-data/vectordb/"
    img_collection_name = "multimodal_collections_images"
    querier = ChromaQuerier(collection_name=img_collection_name, persist_directory=chroma_path)
    
    # 1. Get count
    print(f"Total documents: {querier.get_collection_count()}")
    
    # 2. Basic Query
    results = querier.query_collection(query_texts=["esha"], n_results=5)
    print("Basic Query Results:", results)
    
    # 3. Complex Metadata Query (e.g., category is 'research' AND year >= 2024)
    metadata_filter = {
        "$and": [
            {"category": {"$eq": "research"}},
            {"year": {"$gte": 2024}}
        ]
    }
    filtered_results = querier.query_with_metadata(
        query_texts=["neural networks berkeley"], 
        where_filter=metadata_filter,
        n_results=2
    )
    print("Filtered Metadata Results:", filtered_results)
