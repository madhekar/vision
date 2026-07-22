import chromadb
import json
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

class ChromaQuerier:
    def __init__(self, img_collection: str, vid_collection: str, txt_collection: str, persist_directory: str = "./chroma_db"):
        """Initialize the ChromaDB client and load the collection."""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = OpenCLIPEmbeddingFunction()
        self.img_collection = self.client.get_or_create_collection(name=img_collection, embedding_function=self.embedding_function)
        self.vid_collection = self.client.get_or_create_collection(name=vid_collection, embedding_function=self.embedding_function)
        self.txt_collection = self.client.get_or_create_collection(name=txt_collection, embedding_function=self.embedding_function)


    def get_collection_count(self) -> int:
        """Return the array total number of items in a collection for each modality."""
        return [self.img_collection.count(), self.vid_collection.count(), self.txt_collection.count()]

    def query_image_collection(self, query_texts: list, n_results: int = 2) -> dict:
        """Return semantic similarity search results for given query texts for image collection."""
        return self.img_collection.query(
            query_texts=query_texts,
            n_results=n_results
        )

    def query_video_collection(self, query_texts: list, n_results: int = 2) -> dict:
        """Return semantic similarity search results for given query texts for video collection."""
        return self.vid_collection.query(
            query_texts=query_texts,
            n_results=n_results
        )
    
    def query_text_collection(self, query_texts: list, n_results: int = 2) -> dict:
        """Return semantic similarity search results for given query texts for text collection."""
        return self.txt_collection.query(
            query_texts=query_texts,
            n_results=n_results
        )
    
    def query_with_image_metadata(self, query_texts: list, where_filter: dict, n_results: int = 2) -> dict:
        """Return similarity search results with complex metadata filtering for image collection.
        Example where_filter:
        {"$and": [{"category": {"$eq": "research"}}, {"year": {"$gte": 2024}}]}
        """
        return self.img_collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where_filter
        )

    def query_with_video_metadata(self, query_texts: list, where_filter: dict, n_results: int = 2) -> dict:
        """Return similarity search results with complex metadata filtering for video collection.
        Example where_filter:
        {"$and": [{"category": {"$eq": "research"}}, {"year": {"$gte": 2024}}]}
        """
        return self.vid_collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where_filter
        )
    
    def query_with_text_metadata(self, query_texts: list, where_filter: dict, n_results: int = 2) -> dict:
        """Return similarity search results with complex metadata filtering for text collection.
        Example where_filter:
        {"$and": [{"category": {"$eq": "research"}}, {"year": {"$gte": 2024}}]}
        """
        return self.txt_collection.query(
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
    vid_collection_name = "multimodal_collection_videos"
    txt_collection_name = "multimodal_collection_texts"

    querier = ChromaQuerier(img_collection=img_collection_name, 
                            vid_collection=vid_collection_name,
                            txt_collection=txt_collection_name,
                            persist_directory=chroma_path)
    
    # 1. Get count
    print(f"\n Total modalities per type: \n {querier.get_collection_count()}")
    
    # 2. Basic Query
    results = querier.query_image_collection(query_texts=["esha"], n_results=2)
    
    print("\n Basic Query Results (image collection): \n", json.dumps(results, indent=2))
    
    # 3. Complex Metadata Query (e.g., category is 'research' AND year >= 2024)
    metadata_filter = {
        "$and": [
            {"src": {"$eq": "ASSORT_K30"}},
            {"ts": {"$gte": 946717260}}
        ]
    }
    filtered_results = querier.query_with_image_metadata(
        query_texts=["neural networks berkeley"], 
        where_filter=metadata_filter,
        n_results=2
    )
    print("\n Filtered Metadata Results (image collection): \n", json.dumps(filtered_results, indent=2))

    # 4. Basic Query
    results = querier.query_video_collection(query_texts=["esha"], n_results=2)
    
    print("\n Basic Query Results (video collection): \n", json.dumps(results, indent=2))
    
    # 5. Complex Metadata Query (e.g., category is 'research' AND year >= 2024)
    metadata_filter = {
        "$and": [
            {"src": {"$eq": "ASSORT_K30"}},
            {"ts": {"$gte": 946717260}}
        ]
    }
    filtered_results = querier.query_with_video_metadata(
        query_texts=["neural networks berkeley"], 
        where_filter=metadata_filter,
        n_results=2
    )
    print("\n Filtered Metadata Results (video collection): \n", json.dumps(filtered_results, indent=2))

    # 6. Basic Query
    results = querier.query_text_collection(query_texts=["esha"], n_results=2)
    
    print("\n Basic Query Results (text collection): \n", json.dumps(results, indent=2))
    
    # 7. Complex Metadata Query (e.g., category is 'research' AND year >= 2024)
    metadata_filter = {
        "$and": [
            {"src": {"$eq": "ASSORT_K30"}},
            {"ts": {"$gte": 946717260}}
        ]
    }
    filtered_results = querier.query_with_text_metadata(
        query_texts=["neural networks berkeley"], 
        where_filter=metadata_filter,
        n_results=2
    )
    print("\n Filtered Metadata Results (text collection): \n", json.dumps(filtered_results, indent=2))
