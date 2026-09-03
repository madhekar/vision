import os
import time
import chromadb
from chromadb.config import Settings
import json
import sys
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from  compress_video_helper import compress_video
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
max_bytes = 5 * 1024 * 1024

'''handles chromadb query functions supported by agent'''
def chroma_query_init():
    """Initialize the ChromaDB client and load the collection."""
    client = chromadb.PersistentClient(path="/mnt/zmdata/home-media-app/data/app-data/vectordb/", settings=Settings(anonymized_telemetry=False))
    embedding_function = OpenCLIPEmbeddingFunction()
    img_collection = client.get_or_create_collection(name="multimodal_collection_images", embedding_function=embedding_function)
    vid_collection = client.get_or_create_collection(name="multimodal_collection_videos", embedding_function=embedding_function)
    txt_collection = client.get_or_create_collection(name="multimodal_collection_texts", embedding_function=embedding_function)
    n_results = 2

    return img_collection, vid_collection, txt_collection, n_results

def get_collection_count() -> list[int]:
    img_collection, vid_collection, txt_collection, _ = chroma_query_init()
    """Return the array total number of items in a collection for each modality."""
    return [img_collection.count(), vid_collection.count(), txt_collection.count()]


def query_image_collection( query_texts: list) -> dict:
    """Return semantic similarity search results for given query texts for image collection."""
    img_collection, _, _, n_results = chroma_query_init()
    img_res =  img_collection.query(
        query_texts=query_texts,
        n_results=n_results
    )
    result_list = []
    for ir in img_res["metadatas"][0]:
        result_list.append({"caption": ir["caption"] , "text": ir["text"], "ts": ir["ts"], "uri": ir["uri"] })
    return result_list


def query_video_collection( query_texts: list) -> list:
    """Return semantic similarity search results vuri fields only; for given query texts for video collection ."""
    _, vid_collection, _, n_results = chroma_query_init()
    result = vid_collection.query( query_texts=query_texts, n_results=n_results)
    result_list = []
    for vr in result["metadatas"][0]:
        if os.path.getsize(vr["vuri"]) > max_bytes:
           cvideo = compress_video(vr["vuri"], 20)
           result_list.append({"url": cvideo, "caption": vr["caption"].replace('"', ''), "text": vr["text"], "ts": vr["ts"]})
        else:
           result_list.append({"url": vr["vuri"], "caption": vr["caption"].replace('"', ''), "text": vr["text"], "ts": vr["ts"]})    
    print(result_list)
    return result_list


def query_text_collection( query_texts: list) -> dict:
    """Return semantic similarity search results for given query texts for text collection."""
    _, _, txt_collection, n_results = chroma_query_init()
    return txt_collection.query(
        query_texts=query_texts,
        n_results=n_results
    )

'''
valid src types: Samsung USB, SWEETHOME, GRANDCANYON, Berkeley, ASSORT_K30
'''
def query_image_with_metadata( query_texts: list, src_filter: str, ts_filter_start: int, ts_filter_end: int) -> dict:
    """Return similarity search results with  metadata filtering for image collection."""
    img_collection, _, _, n_results = chroma_query_init()
    metadata_filter = { 
        "$and": [
            {"src": {"$eq": src_filter }},
            {"ts": {"$gte": ts_filter_start }},
            {"ts": {"$lte": ts_filter_end }}  
        ]}
    #print( metadata_filter)
    img_res = img_collection.query(
        query_texts=query_texts,
        n_results = n_results,
        include=["metadatas"],
        where = metadata_filter
    )
    arr = []
    for ir in img_res["metadatas"][0]:
        arr.append({"caption": ir["caption"].replace('"','') , "text": ir["text"], "ts": str(ir["ts"]), "url": ir["uri"]})
    print(arr)
    return arr

def query_video_with_metadata( query_texts: list, src_filter: str, ts_filter_low: int, ts_filter_high: int) -> dict:
    """Return similarity search results with  metadata filtering for video collection."""

    _, vid_collection, _, n_results = chroma_query_init()

    metadata_filter = {
        "$and": [
          {"src": {"$eq": src_filter }},
          {"ts": {"$gte": ts_filter_low }},
          {"ts": {"$lte": ts_filter_high }}  
            ]}
    #print("--->", metadata_filter)

    result = vid_collection.query(
        query_texts=query_texts, 
        n_results=n_results, 
        include=["metadatas"], 
        where=metadata_filter)

    result_list = []
    for vr in result["metadatas"][0]:
        if os.path.getsize(vr["vuri"]) > max_bytes:
           cvideo = compress_video(vr["vuri"], 20)

           result_list.append({"url": cvideo, "caption": vr["caption"].replace('"', ''), "text": vr["text"], "ts": vr["ts"]})
        else:
           result_list.append({"url": vr["vuri"], "caption": vr["caption"].replace('"', ''), "text": vr["text"], "ts": vr["ts"]})    
    print(result_list)
    return result_list


def query_text_with_metadata( query_texts: list, src_filter: str, ts_filter: int) -> dict:
    """Return similarity search results with metadata filtering for text collection."""
    _, _, txt_collection, n_results = chroma_query_init()
    metadata_filter = '{ "$and": [{"src": {"$eq": "' + src_filter + '" }},{"ts": {"$gte":' + str(ts_filter) +'}}]}'
    return txt_collection.query(
        query_texts=query_texts,
        n_results=n_results,
        where=json.loads(metadata_filter)
    )


if __name__=="__main__":
    method_name = sys.argv[1]

    if method_name == "get_collection_count":
        print(get_collection_count())

    elif method_name == "query_image_collection":
        print(query_image_collection(sys.argv[2]))

    elif method_name == "query_video_collection_uri":
        print(query_video_collection(sys.argv[2]))    
        
    elif method_name == "query_text_collection":
        print(query_text_collection(sys.argv[2]))        

    elif method_name == "query_image_with_metadata":
        print(query_image_with_metadata(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])) 

    elif method_name == "query_video_with_metadata":
        print(query_video_with_metadata(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]))

    elif method_name == "query_text_with_metadata":
        print(query_text_with_metadata(sys.argv[2], sys.argv[3], sys.argv[4]))          
