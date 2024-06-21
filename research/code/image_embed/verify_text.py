import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.config import Settings
import os

embedding_function = OpenCLIPEmbeddingFunction()


client = chromadb.PersistentClient(path="DB")

collection_text = client.create_collection(
    name='multimodal_collection_text', 
    embedding_function=embedding_function, 
    )

IMAGE_FOLDER = '/home/madhekar/work/vision/research/code/image_embed/flowers/allflowers'
text_pth = sorted([os.path.join(IMAGE_FOLDER, image_name) for image_name in os.listdir(IMAGE_FOLDER) if image_name.endswith('.txt')])

print('text paths: \n', text_pth)

list_of_text = []
for text in text_pth:
    with open(text, 'r') as f:
        text = f.read()
        list_of_text.append(text)

ids_txt_list = ['id'+str(i) for i in range(len(list_of_text))]

print('text generate ids:\n',ids_txt_list)

collection_text.add(
    documents = list_of_text,
    ids =ids_txt_list
)

results = collection_text.query(
    query_texts=["What is the bellflower?"],
    n_results=1
)

print('query result:', results)

print('count:', collection_text.count())