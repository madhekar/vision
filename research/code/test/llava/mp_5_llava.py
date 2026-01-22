import ollama
import multiprocessing
import os
# !pip install ollama # Ensure you have the library installed

# --- Configuration ---
OLLAMA_HOST_BASE = "http://localhost:"
OLLAMA_PORT_START = 11434 # Default + Base for multiple instances
MODEL_NAME = "llava" # Or your specific LLaVA model

# List of image paths to process
image_paths = [
    '/Users/emadhekar/Pictures/wastushanti.JPG', # Make sure these images exist!
    '/Users/emadhekar/Pictures/alaska_cooking.JPG',
    '/Users/emadhekar/Pictures/ganapti_challe_gavala.jpg',
]

# --- Worker Function ---
def process_image(image_path, server_url):
    """Processes a single image using a specific Ollama server instance."""
    print(f"Processing {os.path.basename(image_path)} on {server_url}...")
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    'role': 'user',
                    'content': 'Describe this image in detail.',
                    'images': [image_path] # LLaVA uses image paths in messages
                }
            ],
            options={'num_predict': 200} # Control response length
        )
        # For LLaVA, the response content is the text description
        description = response['message']['content']
        print(f"Finished {os.path.basename(image_path)}.")
        return (image_path, description)
    except Exception as e:
        print(f"Error processing {image_path} on {server_url}: {e}")
        return (image_path, f"Error: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Start multiple Ollama server instances (in separate terminals/scripts)
    # Example:
    # ollama serve -m llava --host 127.0.0.1:11434 &
    # ollama serve -m llava --host 127.0.0.1:11435 &
    # ollama serve -m llava --host 127.0.0.1:11436 &

    #OLLAMA_HOST=127.0.0.1:11435 ollama serve
    #OLLAMA_HOST=127.0.0.1:11436 ollama serve


    # (Use different ports for each instance if on the same machine)

    # 2. Define server URLs for our workers
    server_urls = [f"{OLLAMA_HOST_BASE}{OLLAMA_PORT_START + i}" for i in range(len(image_paths))]

    # 3. Create a pool of workers
    with multiprocessing.Pool(processes=2) as pool:
        # Map image_paths to their specific server URLs for the worker function
        # We need to pass multiple arguments, so we use starmap or zip arguments
        tasks = [(img_path, url) for img_path, url in zip(image_paths, server_urls)]
        results = pool.starmap(process_image, tasks)

    # 4. Print results
    print("\n--- Results ---")
    for path, description in results:
        print(f"\nImage: {os.path.basename(path)}: {description}")
