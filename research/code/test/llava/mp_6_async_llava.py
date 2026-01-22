import asyncio
from ollama import AsyncClient
import base64
from pathlib import Path

# --- Helper Function to Read Image ---
def encode_image_to_base64(image_path):
    """Reads an image file and encodes it to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            # Read as bytes and encode to base64 for the API
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None

# --- Async Function to Query LLaVA ---
async def analyze_image_llava(image_b64, prompt_text, model_name='llava'):
    """Sends a multimodal request to the LLaVA model using AsyncClient."""
    client = AsyncClient()
    print(f"Processing image with prompt: '{prompt_text[:30]}...'")
    try:
        response = await client.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt_text,
                    'images': [image_b64] # LLaVA expects base64 encoded image data here
                }
            ]
        )
        # Print the response content (can be a stream in other examples)
        print(f"Response for prompt '{prompt_text[:30]}...': {response['message']['content']}")
        return response['message']['content']
    except Exception as e:
        print(f"Error processing image {prompt_text[:30]}...: {e}")
        return None
    finally:
        await client.close() # Important to close the client

# --- Main Async Function to Run the Pool ---
async def main():
    # Assume you have 'image1.jpg' and 'image2.png' in the same directory
    # For demonstration, create dummy base64 images if files don't exist
    # In a real scenario, these would be actual image file paths
    image1_path = "image1.jpg"
    image2_path = "image2.png"

    # Create dummy image files if they don't exist (for testing without real images)
    if not Path(image1_path).exists():
        with open(image1_path, "wb") as f:
            f.write(b"This is a placeholder for image 1 data.")
    if not Path(image2_path).exists():
        with open(image2_path, "wb") as f:
            f.write(b"This is a placeholder for image 2 data.")

    # Encode your actual images
    image1_base64 = encode_image_to_base64(image1_path)
    image2_base64 = encode_image_to_base64(image2_path)

    tasks = [
        await analyze_image_llava(image1_base64, "Describe the main subject in this image."),
        await analyze_image_llava(image2_base64, "What colors are prominent here?"),
        await analyze_image_llava(image1_base64, "Is there any text in the image?"), # Reusing image for demo
    ]

    # Run tasks concurrently (this acts as our pool)
    print("Starting LLaVA image")

if __name__=="__main__":
    asyncio.run(main())
