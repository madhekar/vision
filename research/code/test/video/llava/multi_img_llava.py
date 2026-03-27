import torch
from PIL import Image
import base64
from transformers import AutoProcessor, LlavaForConditionalGeneration

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# 1. Load the model and processor from the Hugging Face Hub
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# 2. Prepare an image and a prompt (replace with your image source)
#image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
raw_image = encode_image_to_base64("/mnt/zmdata/home-media-app/data/input-data/img/GRANDCANYON/00a61654-683e-5315-917d-e2cb7d093a67/vcm_s_kf_m160_160x120.jpg")

#prompt = "USER: <image>\nWhat is shown in this image?\nASSISTANT:"
prompt ="What is shown in this image?"
# 3. Process the inputs
inputs = processor(prompt, raw_image, return_tensors='pt').to(model.device, torch.float16)

# 4. Generate a response
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

# 5. Decode and print the output
print(processor.decode(output[0], skip_special_tokens=True))
