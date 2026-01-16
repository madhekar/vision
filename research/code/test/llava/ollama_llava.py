import ollama
import base64

# Function to encode the image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = '/home/madhekar/temp/filter/training/people/IMG_5379.PNG'
encoded_image = encode_image_to_base64(image_path)

# Perform inference
response = ollama.chat(
    model='llava:13b',
    messages=[
        {
            'role': 'user',
            'content': 'Describe this image in detail:',
            'images': [encoded_image]
        }
    ],
)

print(response['message']['content'])
