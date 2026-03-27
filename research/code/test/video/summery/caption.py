from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
#import requests

# 1. Load the pre-trained processor and model
# The processor handles image preparation and text decoding
# The model performs the actual image-to-text generation
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 2. Define a function to generate the caption
def generate_caption(image_path_or_url):
    """
    Generates a caption for an image using the BLIP model.
    """
    # Load the image from a URL or local file path
    # if image_path_or_url.startswith("http"):
    #     image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert('RGB')
    # else:
    image = Image.open(image_path_or_url).convert('RGB')

    # Prepare the image and an empty text prompt for the model
    # The 'None' prompt tells the model to generate a caption without a text prompt
    inputs = processor(image, text=None, return_tensors="pt")

    # Generate the caption tokens
    # 'max_new_tokens' controls the length of the generated caption
    output_tokens = model.generate(**inputs, max_new_tokens=50)

    # Decode the tokens back into a human-readable string
    caption = processor.decode(output_tokens[0], skip_special_tokens=True)
    return caption

# 3. Example Usage
# You can replace this URL with a local image file path, e.g., "./my_image.jpg"
image_url = "/home/madhekar/Videos/frame0001.jpg" #"https://wikimedia.org"
caption = generate_caption(image_url)

print(f"Image URL/Path: {image_url}")
print(f"Generated Caption: {caption}")
