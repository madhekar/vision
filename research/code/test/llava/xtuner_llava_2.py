import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

"""
The scene displays a computer screen that has been turned upside down. The content of this inverted display is an equation, possibly related to physics or data analysis involving graphs and diagrams.
There are three main sections in each section: one at about 0% on top left corner; another occupying most parts from around mid-height towards bottom right side (with some text overlapping it); 
finally, there's something more centered but not fully visible near middle-right part of frame). 

In addition to these primary components within the central area where you can see both red curve line & blue horizontal bar graph-like element), 
various other elements appear throughout such as arrows pointing upwards along edges like borders between different areas or even vertical lines connecting them together.
 
A few smaller details include two people partially seen close up above center-left portion of screen, presumably observing what appears to be important work being done here! 
And also just below their heads could potentially indicate they're working closely for support purposes too!
  
Overall impression would suggest someone was

"""

model_id = "xtuner/llava-llama-3-8b-transformers"

# torch.cuda.current_device()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prompt = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\nDescribe the image with thoughtful insights using additional information provided<|eot_id|>""<|start_header_id|>assistant<|end_header_id|>\n\n")


model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=False
).to('cpu')

processor = AutoProcessor.from_pretrained(model_id)

img = "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/IMG_5156.PNG"

raw_image = Image.open(img)

inputs = processor(prompt, raw_image, return_tensors='pt').to('cpu', torch.float16)

output = model.generate(**inputs, max_new_tokens=200, repetition_penalty=1.5, temperature=0.5, do_sample=True)
print(processor.decode(output[0][2:], skip_special_tokens=True))