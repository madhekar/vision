import chromadb, torch
import os
from matplotlib import pyplot as plt

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from chromadb.config import Settings
from PIL import Image
import numpy as np


# Get the uris to the images
IMAGE_FOLDER = '/home/madhekar/work/vision/research/code/image_embed/flowers/allflowers'

# vector database persistance 
client = chromadb.Client(Settings(persist_directory='db/'))

#openclip embedding function!
embedding_function = OpenCLIPEmbeddingFunction()

'''
Image collection inside vector database 'chromadb'
'''
image_loader = ImageLoader()

collection_images = client.get_or_create_collection(
    name='multimodal_collection_images', 
    embedding_function=embedding_function, 
    data_loader=image_loader)

# add image embeddings in vector db
image_uris = sorted([os.path.join(IMAGE_FOLDER, image_name) for image_name in os.listdir(IMAGE_FOLDER) if not image_name.endswith('.txt')])
print('=> image urls: \n', '\n'.join(image_uris))
ids = [str(i) for i in range(len(image_uris))]

collection_images.add(ids=ids, uris=image_uris)

'''
text collection inside vector database
'''
collection_text = client.get_or_create_collection(
    name='multimodal_collection_text', 
    embedding_function=embedding_function, 
    )

text_pth = sorted([os.path.join(IMAGE_FOLDER, image_name) for image_name in os.listdir(IMAGE_FOLDER) if image_name.endswith('.txt')])

print('=> text paths: \n', '\n'.join(text_pth))

list_of_text = []
for text in text_pth:
    with open(text, 'r') as f:
        text = f.read()
        list_of_text.append(text)

ids_txt_list = ['id'+str(i) for i in range(len(list_of_text))]

print('=> text generate ids:\n', ids_txt_list)

collection_text.add(
    documents = list_of_text,
    ids =ids_txt_list
)

'''
model autotokenizer and processor componnents for LLM model MC-LLaVA-3b with trust flag
'''
from transformers import AutoTokenizer, AutoProcessor, AutoModel,TextStreamer

model = AutoModel.from_pretrained("visheratin/MC-LLaVA-3b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("visheratin/MC-LLaVA-3b", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("visheratin/MC-LLaVA-3b", trust_remote_code=True)

'''
create query on image, also shows similar document in vector database (not using LLM)
'''
question = 'Answer with organized answers: What type of rose is in the picture? Mention some of its characteristics and how to take care of it ?'
query_image = '/home/madhekar/work/edata/kaggle/input/flowers/flowers/rose/00f6e89a2f949f8165d5222955a5a37d.jpg'
raw_image = Image.open(query_image)

doc = collection_text.query(
    query_embeddings=embedding_function(query_image),
    n_results=1,
)['documents'][0][0]

print('/n=> doccument: ', doc)  

plt.imshow(raw_image)
plt.show()

imgs = collection_images.query(query_uris=query_image, include=['data'], n_results=1)
for img in imgs['data'][0][1:]:
    plt.imshow(img)
    plt.axis("off")
    plt.show()

'''
create prompt to test the LLM
'''
prompt = """<|im_start|>system
A chat between a curious human and an artificial intelligence assistant.
The assistant is an exprt in flowers , and gives helpful, detailed, and polite answers to the human's questions.
The assistant does not hallucinate and pays very close attention to the details.<|im_end|>
<|im_start|>user
<image>
{question} Use the following article as an answer source. Do not write outside its scope unless you find your answer better {article} if you thin your answer is better add it after document.<|im_end|>
<|im_start|>assistant
""".format(question='question', article=doc)

'''
generate propcssor using image and associated prompt query, and generate LLM response
'''
with torch.inference_mode():
    inputs = processor(prompt, [raw_image], model, max_crops=100, num_tokens=728)

streamer = TextStreamer(processor.tokenizer)    

with torch.inference_mode():
  output = model.generate(**inputs, max_new_tokens=200, do_sample=True, use_cache=False, top_p=0.9, temperature=0.2, eos_token_id=processor.tokenizer.eos_token_id, streamer=streamer)


print('=> output:', processor.tokenizer.decode(output[0]).replace(prompt, "").replace("<|im_end|>", ""))  

'''
'''
