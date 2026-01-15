import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "xtuner/llava-llama-3-8b-v1_1-transformers"

prompt = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat are these?<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n")
image_file = "/home/madhekar/temp/training/people/IMG_5379.PNG"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    dtype=torch.int8, 
    low_cpu_mem_usage=True, 
).to(0)

processor = AutoProcessor.from_pretrained(model_id)


raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.int8)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
# These are two cats, one brown and one gray, lying on a pink blanket. sleep. brown and gray cat sleeping on a pink blanket.
