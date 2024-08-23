import chromadb as cdb
import os
from dotenv import load_dotenv

# vector database path
load_dotenv('/home/madhekar/.env.local')
storage_path = os.getenv('STORAGE_PATH')

from chromadb.config import Settings

client = cdb.PersistentClient( path=storage_path, settings=Settings(allow_reset=True))
col = client.get_collection("multimodal_collection_images")

print(col.count())