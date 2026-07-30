import chromadb
import json
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
#from openclaw import Skill, Intent


class chroma_query_executor:
    '''handles chromadb query functions supported by agent'''
    def __init__(self, 
                 persist_directory: str = "/mnt/zmdata/home-media-app/data/app-data/vectordb/", 
                 img_collection: str = "multimodal_collection_images", 
                 vid_collection: str = "multimodal_collection_videos", 
                 txt_collection: str = "multimodal_collection_texts", 
                 n_results: int = 2
                 ):
        """Initialize the ChromaDB client and load the collection."""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = OpenCLIPEmbeddingFunction()
        self.img_collection = self.client.get_or_create_collection(name=img_collection, embedding_function=self.embedding_function)
        self.vid_collection = self.client.get_or_create_collection(name=vid_collection, embedding_function=self.embedding_function)
        self.txt_collection = self.client.get_or_create_collection(name=txt_collection, embedding_function=self.embedding_function)
        self.n_results = n_results

    def get_collection_count(self) -> list[int]:
        """Return the array total number of items in a collection for each modality."""
        return [self.img_collection.count(), self.vid_collection.count(), self.txt_collection.count()]


    def query_image_collection(self, query_texts: list) -> dict:
        """Return semantic similarity search results for given query texts for image collection."""
        img_res =  self.img_collection.query(
            query_texts=query_texts,
            n_results=self.n_results
        )
        arr = []
        for ir in img_res["metadatas"][0]:
           arr.append({"caption": ir["caption"] , "description": ir["text"]})
        return arr

    def query_video_collection(self, query_texts: list) -> dict:
        """Return semantic similarity search results for given query texts for video collection."""
        return self.vid_collection.query(
            query_texts=query_texts,
            n_results=self.n_results
        )
    

    def query_text_collection(self, query_texts: list) -> dict:
        """Return semantic similarity search results for given query texts for text collection."""
        return self.txt_collection.query(
            query_texts=query_texts,
            n_results=self.n_results
        )

    def query_with_image_metadata(self, query_texts: list, src_filter: str, ts_filter: int) -> dict:
        """Return similarity search results with  metadata filtering for image collection."""
        metadata_filter = '{ "$and": [{"src": {"$eq": "' + src_filter + '" }},{"ts": {"$gte":' + str(ts_filter) +'}}]}'
        return self.img_collection.query(
            query_texts=query_texts,
            n_results=self.n_results,
            where= json.loads(metadata_filter)
        )
    

    def query_with_video_metadata(self, query_texts: list, src_filter: str, ts_filter: int) -> dict:
        """Return similarity search results with  metadata filtering for video collection."""
        metadata_filter = '{ "$and": [{"src": {"$eq": "' + src_filter + '" }},{"ts": {"$gte":' + str(ts_filter) +'}}]}'
        return self.vid_collection.query(
            query_texts=query_texts,
            n_results=self.n_results,
            where=json.loads(metadata_filter)
        )
    

    def query_with_text_metadata(self, query_texts: list, src_filter: str, ts_filter: int) -> dict:
        """Return similarity search results with metadata filtering for text collection."""
        metadata_filter = '{ "$and": [{"src": {"$eq": "' + src_filter + '" }},{"ts": {"$gte":' + str(ts_filter) +'}}]}'
        return self.txt_collection.query(
            query_texts=query_texts,
            n_results=self.n_results,
            where=json.loads(metadata_filter)
        )

