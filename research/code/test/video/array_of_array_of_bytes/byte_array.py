import ollama
import base64
from io import BytesIO
from PIL import Image

# 1. Create or load an image in memory (e.g., using PIL)
img = Image.new('RGB', (100, 100), color = 'red')
buffered = BytesIO()
img.save(buffered, format="PNG")

# 2. Convert raw bytes to base64 string
img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

# 3. Pass the base64 string to the model
response = ollama.chat(model='llava:7b-v1.6', messages=[{
    'role': 'user',
    'content': 'What color is this image?',
    'images': [img_base64] # Pass base64 string here
}])

print(response.message.content)
