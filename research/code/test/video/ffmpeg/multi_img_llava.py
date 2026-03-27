#import requests
from PIL import Image
import torch
#from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
'''
pip install Pillow
pip install torch torchvision torchaudio torchcodec av
pip install accelerate
pip install -U git+https://github.com/huggingface/transformers.git

'''

from transformers import AutoProcessor, LlavaForConditionalGeneration

# Load the model and processor in half-precision
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", 
    dtype=torch.float16, 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Load model and processor
# model_id = "lmms-lab/llama3-llava-next-8b"
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
# processor = AutoProcessor.from_pretrained(model_id)

# Load images
url1 = "/home/madhekar/Videos/ffmpeg_frames/video_1/frames/anjali_ind_tour001.png"
url2 = "/home/madhekar/Videos/ffmpeg_frames/video_1/frames/anjali_ind_tour002.png"
images = [Image.open(url).convert('RGB') for url in [url1, url2]]

# Prepare input: ensure <image> tokens match image count
prompt = "<image><image>\nDescribe the two images."
messages = [{"role": "user", "content": prompt}]
prompt_templated = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# Process and generate
inputs = processor(text=prompt_templated, images=images, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.batch_decode(output, skip_special_tokens=True)[0])
