import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load Model and Processor
# Using LLaVA-NeXT (1.6) Mistral 7B
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

# Load in 16-bit for better memory efficiency
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

# 3. Prepare Image and Prompt
# Example image from Hugging Face
url = "/home/madhekar/temp/filter/training/people/IMG_5379.PNG"


image = Image.open(url)

# Proper format for LLaVA-NeXT: "USER: <image>\n<prompt>\nASSISTANT:"
prompt = "USER: <image>\nWhat is this picture? ASSISTANT:"

# 4. Preprocess Input
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

# 5. Generate Output
output = model.generate(**inputs, max_new_tokens=200)

# 6. Decode and Print Result
print(processor.decode(output[0], skip_special_tokens=True))
