import chromadb as cdb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings, DEFAULT_TENANT

def init_vdb(vdp, icn, tcn, vcn):
    # vector database persistance
    client = cdb.PersistentClient( path=vdp, tenant=DEFAULT_TENANT, settings=Settings(allow_reset=False))
    
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
