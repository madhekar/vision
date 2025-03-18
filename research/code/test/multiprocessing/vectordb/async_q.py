import asyncio
import time
import json
import glob
import pandas as pd

# import util
import os
import uuid
import chardet
import chromadb as cdb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction as stef
from aiomultiprocess import Pool

client = cdb.PersistentClient(path='./', settings=Settings(allow_reset=True))

def fileList(path, pattern="**/*", recursive=True):
    files = glob.glob(os.path.join(path, pattern), recursive=recursive)
    return files

def create_chromadb(path, text_collection_name):
    #client = cdb.PersistentClient( path=path, settings=Settings(allow_reset=True))

    # reset chromadb persistant store
    # client.reset()

    # list of collections
    collections_list = [c.name for c in client.list_collections()]

    print(f'->>{collections_list}')

    # openclip embedding function!
    embedding_function = stef()

    collection_text = client.get_or_create_collection(
        name=text_collection_name,
        embedding_function=embedding_function,
    )

async def populate_vdb(text_collection, furi):
    await asyncio.sleep(1)
    return x * x


async def main():
    vdb_path = './'
    text_conllection_name = 'multimodal_docs'
    start_time = time.time()

    file_list= fileList()
    create_chromadb(vdb_path, text_conllection_name )
    
    async with Pool(4) as pool:
        results = await pool.map(populate_vdb, range(10))
        print(results)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
