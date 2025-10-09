from transformers import AutoProcessor, LlavaForConditionalGeneration
from huggingface_hub import snapshot_download
from PIL import Image
import requests
import bitsandbytes
import accelerate
import modelbit

def prompt_llava(url: str, prompt: str):
    image = Image.open(requests.get(url, stream=True).raw)
    mb.log_image(image)  # Log the input image in Modelbit
    full_prompt = f"USER: <image>\n{prompt} ASSISTANT:"
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to("cuda")
    generate_ids = model.generate(**inputs, max_new_tokens=15)
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].split("ASSISTANT:")[1]
    return response