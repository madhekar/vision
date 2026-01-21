import ollama
import multiprocessing
import base64
import json
import os # For checking Ollama server

# --- Configuration ---
OLLAMA_HOST = 'http://localhost:11434'
MODEL_NAME = 'llava' # Or 'llava:7b', etc.
NUM_WORKERS = 2 # Number of concurrent processes

# --- Helper Functions ---

def load_image_base64(image_path):
    """Encodes an image file to a base64 string."""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def chat_with_ollama(prompt_data):
    """Function for a single process to interact with Ollama."""
    try:
        # Construct the chat payload
        messages = [
            {"role": "system", "content": "You are a helpful assistant that describes images and answers questions about them."},
            {"role": "user", "content": prompt_data["text"], "images": [prompt_data["image_base64"]]}
        ]
        # Use the ollama Python library
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={'temperature': 0.7},
            stream=False # Get full response at once
        )
        return {
            "status": "success",
            "response": response['message']['content'],
            "prompt_id": prompt_data['id']
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "prompt_id": prompt_data['id']
        }

# --- Main Execution ---

def main():
    # Check if Ollama server is running (basic check)
    try:
        list(ollama.list()) # A simple way to check connectivity
    except Exception as e:
        print(f"Error connecting to Ollama at {OLLAMA_HOST}. Is it running? {e}")
        return

    # Sample prompts (replace with your actual image/text pairs)
    # Make sure 'image1.jpg', 'image2.png' exist or adjust paths
    sample_prompts = [
        {"id": 1, "text": "Describe this image.", "image_base64": load_image_base64("image1.jpg")},
        {"id": 2, "text": "What is happening in this picture?", "image_base64": load_image_base64("image2.png")},
    ]