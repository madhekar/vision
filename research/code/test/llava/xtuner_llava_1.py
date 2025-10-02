from transformers import pipeline
from PIL import Image    
import requests

model_id = "xtuner/llava-llama-3-8b-v1_1-transformers"
pipe = pipeline("image-to-text", model=model_id, device=0)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"

image = Image.open(requests.get(url, stream=True).raw)
prompt = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat are these?<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n")
outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)
[{'generated_text': 'user\n\n\nWhat are these?assistant\n\nThese are two cats, one brown and one gray, lying on a pink blanket. sleep. brown and gray cat sleeping on a pink blanket.'}]