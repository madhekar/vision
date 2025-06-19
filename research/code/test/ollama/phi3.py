# from transformers import AutoProcessor, LlavaForConditionalGeneration
# import requests
# from PIL import Image

# # Load the processor and model
# processor = AutoProcessor.from_pretrained("llava-hf/llava-phi-3-mini-4k-instruct")
# model = LlavaForConditionalGeneration.from_pretrained(
#     "llava-hf/llava-phi-3-mini-4k-instruct"
# )

# # Prepare the image and text
# image_url = "/home/madhekar/work/home-media-app/data/input-data/img/20130324-3I3A4652-X2.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw)

# prompt = "USER: <image>\nWhat is this image?\nASSISTANT:"

# # Process inputs
# inputs = processor(text=prompt, images=image, return_tensors="pt")

# # Generate text
# output = model.generate(**inputs, max_new_tokens=100)

# # Decode and print the output
# print(processor.decode(output[0], skip_special_tokens=True))



from transformers import pipeline
from PIL import Image    

model_id = "xtuner/llava-phi-3-mini-hf"
pipe = pipeline("image-to-text", model=model_id, device="cpu")
url = '/home/madhekar/work/home-media-app/data/input-data/img/Grad Pics family6.jpg'
#"/home/madhekar/work/home-media-app/data/input-data/img/20130324-3I3A4652-X2.jpg"

#image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(url)
prompt = "<|user|>\n<image>\nCan you describe image in detail?<|end|>\n<|assistant|>\n"

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)

