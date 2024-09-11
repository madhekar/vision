import chromadb as cdb
import os
from dotenv import load_dotenv

# vector database path
load_dotenv("/home/madhekar/.env.local")
storage_path = os.getenv("STORAGE_PATH")
image_collection = "multimodal_collection_images"

client = cdb.PersistentClient( path=storage_path, settings=cdb.config.Settings(allow_reset=True))
col = client.get_collection(image_collection)

count = col.count()
print(count)

batch_size = 1

for i in range(0, count, batch_size):
    batch = col.get(
        include=[
            "uris",
            "metadatas",
        ],  # , "embeddings","data","distances","uris"],
        limit=batch_size,
        offset=i
    )
    #print("batchid:",i , batch)

""" res = col.query(
    query_images="",
    n_results=1,
    where={"ids" : {"$eq": "fcce570d-a85b-453e-876d-2af6400ce919"}}
)    

print(res) """

print(col.get(where={"ids": {"$eq" : "ff842096-9853-4453-a50b-278c7ad19401"}}))

print(col.get("ff842096-9853-4453-a50b-278c7ad19401"))  # works!

col.update(ids="ff842096-9853-4453-a50b-278c7ad19401", metadatas={"names":"Anjali,Shoma,Esha"})

print(col.get("ff842096-9853-4453-a50b-278c7ad19401"))  # works!