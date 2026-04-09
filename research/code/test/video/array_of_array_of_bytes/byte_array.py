import ollama
import base64
from io import BytesIO
from PIL import Image

# 1. Create or load an image in memory (e.g., using PIL)
img = Image.new('RGB', (100, 100), color = 'red')
buffered = BytesIO()
img.save(buffered, format="PNG")

img2 = Image.new('RGB', (100, 100), color = 'green')
buffered2 = BytesIO()
img.save(buffered, format="PNG")

# 2. Convert raw bytes to base64 string
img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

img_base64_2 = base64.b64encode(buffered.getvalue()).decode('utf-8')

# 3. Pass the base64 string to the model
response = ollama.chat(
    model='mapler/llama3-llava-next-8b:latest', 
    messages=[{
    'role': 'user',
    'content': 'What color is this images?',
    'images': [img_base64, img_base64_2] # Pass base64 string here
}])

print(response.message.content)
