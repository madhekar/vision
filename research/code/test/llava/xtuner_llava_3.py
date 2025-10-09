from transformers import AutoProcessor, LlavaForConditionalGeneration
from huggingface_hub import snapshot_download
from PIL import Image
import requests
import bitsandbytes
import accelerate
import modelbit
import torch

"""
!pip install accelerate bitsandbytes transformers Pillow modelbit
"""

def prompt_llava(url: str, prompt: str):
    image = Image.open(requests.get(url, stream=True).raw)
    #mb.log_image(image)  # Log the input image in Modelbit
    full_prompt = f"USER: <image>\n{prompt} ASSISTANT:"
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to("cpu")
    generate_ids = model.generate(**inputs, max_new_tokens=15)
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].split("ASSISTANT:")[1]
    return response

def load_model():
    model_id = "xtuner/llava-llama-3-8b-transformers"

    with modelbit.setup(name="load_model"):
      model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                        local_files_only=True,
                                                        load_in_8bit=True,
                                                        device_map='auto',
                                                        torch_type=torch.float32)
      processor = AutoProcessor.from_pretrained(model_id, local_files_only=True, load_in_8bit=True).to('cpu')

    return model, processor

processor, model = load_model()

prompt_llava("https://doc.modelbit.com/img/cat.jpg", "What animals are in this picture?")

modelbit.deploy(
    prompt_llava,
    extra_files=["llava-hf"],
    python_packages=["bitsandbytes==0.44.3", "accelerate==1.10.0"],
    setup="load_model",
    require_gpu=False,
)