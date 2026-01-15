from transformers import pipeline
import torch
from PIL import Image    
import os

model_id = "xtuner/llava-llama-3-8b-v1_1-transformers"
pipe = pipeline("image-to-text", model=model_id, device=0)

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()

torch.cuda.memory_summary(device=0, abbreviated=False)
image = Image.open("/home/madhekar/temp/training/people/IMG_5379.PNG")
prompt = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat are these?<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n")
outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)
# [{'generated_text': 'user\n\n\nWhat are these?assistant\n\nThese are two cats, one brown and one gray, lying on a pink blanket. sleep. brown and gray cat sleeping on a pink blanket.'}]
