import multiprocessing
from multiprocessing import Manager, Process

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import time


 #, settings=Settings(chroma_db_impl="duckdb+parquet", allow_reset=True))

def process_data(data_queue, collection_name):

    client = chromadb.PersistentClient(path='./vdb', settings=Settings(allow_reset=True))

    default_ef = embedding_functions.SentenceTransformerEmbeddingFunction()

    collection = client.get_collection(name=collection_name, embedding_function=default_ef)

    while True:
        item = data_queue.get()
        if item is None:
            break
        collection.add(documents=[item['document']], ids=[item['id']])
        data_queue.task_done()
    print(f"Worker finished processing for {collection_name}")

def main():
            
    collection_name = "my_collection"
    client = chromadb.PersistentClient(
        path="./vdb", settings=Settings(allow_reset=True)
    )

    client.reset()

    default_ef = embedding_functions.SentenceTransformerEmbeddingFunction()

    try:
        client.get_collection(name=collection_name)
        client.delete_collection(name=collection_name)
    except Exception as err:
        print(err)

    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=default_ef
    )

    with Manager() as manager:

        data_queue = manager.Queue()
        
        # Sample data
        data = [
            {"id": "1", "document": "This is document 1."},
            {"id": "2", "document": "This is document 2."},
            {"id": "3", "document": "This is document 3."},
            {"id": "4", "document": "This is document 4."},
            {"id": "5", "document": "This is document 5."},
        ]

        # Initialize and start worker processes
        num_processes = 2 #multiprocessing.cpu_count()
        processes = []
        for _ in range(num_processes):
            process = Process(target=process_data, args=(data_queue, collection_name))
            processes.append(process)
            process.start()

        # Populate queue with data
        for item in data:
            data_queue.put(item)
        print('-->> added items in q')

        # Add sentinel values to signal the end of processing
        for _ in range(num_processes):
            data_queue.put(None)
        print('-->> data put')

        # Wait for all tasks to be processed
        data_queue.join()
        print("-->> joined in q")

        # Wait for all worker processes to finish
        # for process in processes:
        #     process.close()
        #     process.join()

        print("All workers finished")

        # for process in processes:
        #     process.close()

        # Perform a query to verify data is in ChromaDB
        client = chromadb.Client()

        collection = client.get_collection(name=collection_name)

        print(client.count_collections())

        results = collection.query(query_texts=["document"], n_results=2)
        print(results)

if __name__ == "__main__":
    main()
