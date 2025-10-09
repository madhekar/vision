import torch
from PIL import Image
import requests

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)

# Define the quantization configuration
#quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) # Use float16 for faster computation

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)

# Specify the LLaVA model ID
model_id =  "llava-hf/llava-1.5-7b-hf" # Or another version like llava-v1.6-vicuna-7b-hf "xtuner/llava-llama-3-8b-transformers"

# Load the model and processor from Hugging Face, applying the quantization
# processor = AutoProcessor.from_pretrained(model_id)
# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     quantization_config=quantization_config,
#     device_map="auto",  # Let Accelerate handle device mapping
# )

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_4bit=True, quantization_config=quantization_config)

print(model.get_memory_footprint())
# Load an image
image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# Prepare the prompt and the image for the model
prompt = "What is shown in this image?"
inputs = model(text=prompt, images=image, return_tensors="pt").to("cpu")

# Generate the response
output = model.generate(**inputs, max_new_tokens=200)

# Decode and print the result
print(model.decode(output[0], skip_special_tokens=True))