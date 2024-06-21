import chromadb
import os

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings

from matplotlib import pyplot as plt

client = chromadb.PersistentClient(path="DB")

embedding_function = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()

collection_images = client.create_collection(
    name='multimodal_collection_images', 
    embedding_function=embedding_function, 
    data_loader=image_loader)

# Get the uris to the images
IMAGE_FOLDER = '/home/madhekar/work/vision/research/code/image_embed/flowers/allflowers'


image_uris = sorted([os.path.join(IMAGE_FOLDER, image_name) for image_name in os.listdir(IMAGE_FOLDER) if not image_name.endswith('.txt')])
print('\n', image_uris)
ids = [str(i) for i in range(len(image_uris))]

collection_images.add(ids=ids, uris=image_uris)

from matplotlib import pyplot as plt

retrieved = collection_images.query(query_texts=["tulip"], include=['data'], n_results=3)
for img in retrieved['data'][0]:
    plt.imshow(img)
    plt.axis("off")
    plt.show()

retrieved = collection_images.query(query_texts=["bellflower"], include=['data'], n_results=1)
for img in retrieved['data'][0]:
    plt.imshow(img)
    plt.axis("off")
    plt.show()

from PIL import Image
import numpy as np

query_image = np.array(Image.open(f"/home/madhekar/work/edata/kaggle/input/flowers/flowers/daisy/0.jpg"))
print("Query Image")
plt.imshow(query_image)
plt.axis('off')
plt.show()

print("Results")
retrieved = collection_images.query(query_images=[query_image], include=['data'], n_results=3)
for img in retrieved['data'][0][1:]:
    plt.imshow(img)
    plt.axis("off")
    plt.show()

query_image = np.array(Image.open(f"/home/madhekar/work/edata/kaggle/input/flowers/flowers/rose/0444a369fb.jpg"))
print("Query Image")
plt.imshow(query_image)
plt.axis('off')
plt.show()

print("Results")
retrieved = collection_images.query(query_images=[query_image], include=['data'], n_results=1)
for img in retrieved['data'][0][1:]:
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    