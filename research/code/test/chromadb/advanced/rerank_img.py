
from sentence_transformers import CrossEncoder
from PIL import Image
import numpy as np
import torch
import chromadb as cdb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings, DEFAULT_TENANT

def init_vdb(vdp, icn, tcn, vcn):
    # vector database persistance
    client = cdb.PersistentClient( path=vdp, tenant=DEFAULT_TENANT ,settings=Settings(allow_reset=False))
    
    # openclip embedding function!
    embedding_function = OpenCLIPEmbeddingFunction()
    #text_embedding_function = OpenCLIPEmbeddingFunction(model_name="coca_roberta-ViT-B-32") 

    # Image collection inside vector database 'chromadb'
    image_loader = ImageLoader()

    # collection images defined
    collection_images = client.get_or_create_collection(
      name=icn, 
      embedding_function=embedding_function, 
      metadata={"hnsw:space": "cosine"},
      data_loader=image_loader
      )
    
    collection_videos = client.get_or_create_collection(
           name=vcn,
           embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
            data_loader=image_loader,
        )
    
    #Text collection inside vector database 'chromadb'
    collection_text = client.get_or_create_collection(
      name=tcn,
      metadata={"hnsw:space": "cosine"},
      embedding_function=embedding_function,
    )

    #print("*****", collection_text.peek())
    return client, collection_images, collection_text, collection_videos

def rerank_image_search(img_url, image_collection, txt_collection, video_collection):

        # Cross-encoder for precise reranking 
        # (You can use a cross-encoder trained on image-text tasks or text if your query is text-based)
        reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Retrieve top 20 candidate images from ChromaDB
        results = image_collection.query(
            query_uris=img_url,
            include=["uris", "documents", "embeddings", "metadatas"],
            n_results=50,
        )

        # Extract candidates
        candidate_uris = results["uris"][0]
        #candidate_embeddings = results["embeddings"][0]
        d = dict(zip(results["uris"][0], results["metadatas"][0]))
        #print(f"mapping {d}")

        # 4. Query Phase - Stage 2: Deep Reranking (Cross-Encoder)
        # Create pairs: [Query Image, Candidate Image] for the cross-encoder to score
        pairs = []
        for uri in candidate_uris:
            # Load candidate image to pair with the query image
            pairs.append([img_url, uri])

        # Predict relevance scores
        scores = reranker_model.predict(pairs)

        # 5. Sort and Display Top K Results
        # Combine URIs and their scores, then sort by relevance
        scored_results = list(zip(candidate_uris, scores))
        reranked_results = sorted(scored_results, key=lambda x: x[1], reverse=True)

        # Print top 5 reranked images
        top_k = 10
        for i, (uri, score) in enumerate(reranked_results[:top_k]):
            print(f"Rank {i+1} | Image: {uri} | Cross-Encoder Score: {score:.4f} | caption: {d[uri]['caption']}")


i_url = "/home/madhekar/Pictures/IMGP3280.JPG"

vdp="/mnt/zmdata/home-media-app/data/app-data/vectordb"
icn="multimodal_collection_images"
tcn="multimodal_collection_texts"
vcn="multimodal_collection_videos"
client, img_c, txt_c, vid_c = init_vdb( vdp, icn, tcn, vcn)
rerank_image_search(img_url=i_url, image_collection=img_c, txt_collection=txt_c, video_collection=vid_c )