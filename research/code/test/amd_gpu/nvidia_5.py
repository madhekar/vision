from PIL import Image
import requests
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

# Configuration: Choose a model ID (e.g., "llava-hf/llava-1.5-7b-hf").
model_id = "llava-hf/llava-1.5-7b-hf"

# Optional: Configure 4-bit quantization for lower VRAM GPUs.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Load Model and Processor: The device_map="auto" places the model on the available GPU.
print(f"Loading model: {model_id}...")
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16
)
print("Model loaded successfully.")

# Prepare Image: Example image URL is used.
# url = "https://github.com"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("/home/madhekar/temp/training/people/IMG_5379.PNG")
# Define Prompt: LLaVA uses a specific chat template.
prompt = "What is shown in this image? Provide a detailed description."
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device, torch.float16)

# Generate and Decode Output:
print("Generating response...")
output_tokens = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(output_tokens[0], skip_special_tokens=True)
print("\n--- Model Response ---")
print(response)
