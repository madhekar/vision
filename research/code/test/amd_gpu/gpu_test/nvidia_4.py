from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
from PIL import Image

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

pipe_4bit = pipeline(
     model= "llava-hf/llava-v1.6-mistral-7b-hf", #"xtuner/llava-llama-3-8b-v1_1-transformers", #"facebook/opt-1.3b", 
     model_kwargs={"quantization_config":bnb_config},
     device_map="auto" # Automatically map to available devices
)

torch.cuda.empty_cache()

torch.cuda.memory_summary(device=0, abbreviated=False)

image = Image.open("/home/madhekar/temp/filter/training/people/IMG_5379.PNG")

prompt_4bit = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat are these?<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n")

#outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

#print(outputs)
#prompt_4bit="4 bit quantized model"

out = pipe_4bit(prompt_4bit, do_sample=False, top_p=0.95, max_new_tokens=50)

print(out[0]['generated_text'])