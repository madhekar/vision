import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from PIL import Image
import numpy as np
from PIL import Image
import torch
import chromadb as cdb
from sentence_transformers import CrossEncoder
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
        # 1. Initialize Models & ChromaDB
        # Bi-encoder model for fast image/text embedding (using CLIP here for multimodal support)
        #embedding_model = SentenceTransformer("clip-ViT-B-32")

        # Cross-encoder for precise reranking 
        # (You can use a cross-encoder trained on image-text tasks or text if your query is text-based)
        reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # client = chromadb.Client()
        # collection = client.get_or_create_collection(name="image_store")

        # 2. Add Images to ChromaDB 
        # (Assuming you already converted your images to vector embeddings)
        # image_embeddings = embedding_model.encode([Image.open("img1.jpg"), Image.open("img2.jpg")]).tolist()
        # collection.add(
        #     embeddings=image_embeddings,
        #     uris=["img1.jpg", "img2.jpg"], # Storing file paths
        #     ids=["id1", "id2"]
        # )

        # 3. Query Phase - Stage 1: Fast Retrieval (Bi-Encoder)
        #query_image = Image.open(img_url)
        #query_embedding = embedding_model.encode(query_image).tolist()

        # Retrieve top 20 candidate images from ChromaDB
        results = image_collection.query(
            #query_embeddings=[query_embedding],
            query_uris=img_url,
            n_results=50,
            include=["uris", "documents", "embeddings", "metadatas"]
        )

        # Extract candidates
        candidate_uris = results["uris"][0]
        candidate_embeddings = results["embeddings"][0]

        # 4. Query Phase - Stage 2: Deep Reranking (Cross-Encoder)
        # Create pairs: [Query Image, Candidate Image] for the cross-encoder to score
        pairs = []
        for uri in candidate_uris:
            # Load candidate image to pair with the query image
            #candidate_img = Image.open(uri) 
            #pairs.append([query_image, candidate_img])
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
            print(f"Rank {i+1} | Image: {uri} | Cross-Encoder Score: {score:.4f}")


i_url = "/home/madhekar/Pictures/IMGP3280.JPG"

vdp="/mnt/zmdata/home-media-app/data/app-data/vectordb"
icn="multimodal_collection_images"
tcn="multimodal_collection_texts"
vcn="multimodal_collection_videos"
client, img_c, txt_c, vid_c = init_vdb( vdp, icn, tcn, vcn)
rerank_image_search(img_url=i_url, image_collection=img_c, txt_collection=txt_c, video_collection=vid_c )