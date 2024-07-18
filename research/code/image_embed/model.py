'''
from transformers import AutoModel, AutoProcessor
import torch

model = AutoModel.from_pretrained("visheratin/MC-LLaVA-3b", torch_dtype=torch.float16, trust_remote_code=True).to("cuda")

processor = AutoProcessor.from_pretrained("visheratin/MC-LLaVA-3b", trust_remote_code=True)

with torch.inference_mode():
    inputs = processor(prompt, [raw_image], model, max_crops=100, num_tokens=728)
    output = model.generate(**inputs, max_new_tokens=200, use_cache=True, do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.eos_token_id)

result = processor.tokenizer.decode(output[0]).replace(prompt, "").replace("<|im_end|>", "")
print(result)
'''
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModel,TextStreamer
from PIL import Image
import requests
import matplotlib.pyplot as plt


model = AutoModel.from_pretrained("visheratin/MC-LLaVA-3b")
tokenizer = AutoTokenizer.from_pretrained("visheratin/MC-LLaVA-3b")
processor = AutoProcessor.from_pretrained("visheratin/MC-LLaVA-3b", trust_remote_code=True)

image_file = "/home/madhekar/work/edata/kaggle/input/flowers/flowers/daisy/0662daef93.jpg"
raw_image = Image.open(image_file)
plt.imshow(raw_image)
plt.show()
'''

'''


'''

'''
question = 'Answer with organized answers: What type of rose is in the picture? Mention some of its characteristics and how to take care of it ?'
query_image = '/home/madhekar/work/edata/kaggle/input/flowers/flowers/rose/00f6e89a2f949f8165d5222955a5a37d.jpg'
raw_image = Image.open(query_image)

prompt = """<|im_start|>user
<image>
Describe the image.<|im_end|>
<|im_start|>assistant
"""

with torch.inference_mode():
    inputs = processor(prompt, [raw_image], model, max_crops=100, num_tokens=728)

streamer = TextStreamer(processor.tokenizer)    


with torch.inference_mode():
  output = model.generate(**inputs, max_new_tokens=200, do_sample=True, use_cache=False, top_p=0.9, temperature=1.2, eos_token_id=processor.tokenizer.eos_token_id, streamer=streamer)

print(processor.tokenizer.decode(output[0]).replace(prompt, "").replace("<|im_end|>", ""))  

""" doc = collection_text.query(
    query_embeddings=embedding_function(query_image),
    
    n_results=1,
        
)['documents'][0][0]

plt.imshow(raw_image)
plt.show()
imgs = collection_images.query(query_uris=query_image, include=['data'], n_results=3)
for img in imgs['data'][0][1:]:
    plt.imshow(img)
    plt.axis("off")
    plt.show() """