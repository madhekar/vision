import chromadb
from PIL import Image
import torch
import chromadb as cdb
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings, DEFAULT_TENANT
'''
vectordb:
  vectordb_path: /mnt/zmdata/home-media-app/data/app-data/vectordb 
  image_collection_name: multimodal_collection_images
  text_collection_name: multimodal_collection_texts
  video_collection_name: multimodal_collection_videos
  audio_collection_name: multimedia_collection_audios

'''
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

def extract_reranking_results(img_url, image_collection, txt_collection, video_collection):
    # 1. Initialize ChromaDB Client & Collection (assumes embeddings are generated via CLIP/OpenCLIP)
    # client = chromadb.Client()
    # collection = client.create_collection(name="multimodal_images")


    # 2. Add sample images (Placeholder for your data)
    # Note: You would normally add your image URIs or base64 strings to ChromaDB metadata 
    # and their corresponding multimodal embeddings to the embedding column.
    # collection.add(
    #     embeddings=[[...]], # Multimodal vector
    #     metadatas=[{"image_path": "dog_park.jpg"}, {"image_path": "city_street.jpg"}],
    #     ids=["img1", "img2"]
    # )

    # 3. Retrieve Top-K Candidates from ChromaDB
    # For this example, assume we are querying with the text "A dog playing in the park"
    #query_embedding = [...] # Vector representing your text or image query
    results = image_collection.query(
        query_uris=img_url,
        #query_embeddings=[query_embedding],
        include=["data", "metadatas"],
        n_results=50 # Retrieve more than you need, to be narrowed down by the reranker
    )

    # 4. Extract candidates and prepare for reranking
    candidate_images = results['metadatas'][0] 
    candidate_ids = results['ids'][0]

    # 5. Initialize a Vision-Language or Cross-Encoder model
    # For image-to-text or text-to-image ranking, a strong cross-encoder/VLM is required.
    # In this example, we use a SentenceTransformer cross-encoder for semantic pairs.
    # For true multimodal scoring, you might swap this for a VLM like BLIP or CLIP-Score evaluation.
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # 6. Format pairs for the reranker (Query vs Candidate descriptions)
    query_text = "Esha performing Bharatnatyam dance"
    pairs = [[query_text, meta['text']] for meta in candidate_images]

    # 7. Compute exact relevance scores and Rerank
    scores = reranker.predict(pairs)

    # Attach scores to the results and sort them
    for i, meta in enumerate(candidate_images):
        meta['score'] = scores[i]

    # Sort by highest relevance score
    reranked_results = sorted(candidate_images, key=lambda x: x['score'], reverse=True)

    # Output the Top 3 reranked image paths
    print("Top Reranked Images:")
    for img in reranked_results[:10]:
        print(img)
        #print(f"Path: {img['image_path']} | Score: {img['score']:.4f}")


i_url = "/home/madhekar/Pictures/IMGP3280.JPG"

vdp="/mnt/zmdata/home-media-app/data/app-data/vectordb"
icn="multimodal_collection_images"
tcn="multimodal_collection_texts"
vcn="multimodal_collection_videos"
client, img_c, txt_c, vid_c = init_vdb( vdp, icn, tcn, vcn)
extract_reranking_results(img_url=i_url, image_collection=img_c, txt_collection=txt_c, video_collection=vid_c )