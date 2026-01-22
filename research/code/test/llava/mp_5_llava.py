import ollama
import multiprocessing
import os
# !pip install ollama # Ensure you have the library installed

# --- Configuration ---
OLLAMA_HOST_BASE = "http://localhost:"
OLLAMA_PORT_START = 11434 # Default + Base for multiple instances
MODEL_NAME = "llava:13b" # Or your specific LLaVA model

# List of image paths to process
image_paths = [
    '/home/madhekar/temp/filter/training/people/IMG_5543.jpeg',
    '/home/madhekar/temp/filter/training/people/neighbors.jpg',
    '/home/madhekar/temp/filter/training/people/IMG_5533.jpeg',
    '/home/madhekar/temp/filter/training/people/IMG_7285.jpeg',
    '/home/madhekar/temp/filter/training/people/IMG_7460.jpeg'


    # '/Users/emadhekar/Pictures/wastushanti.JPG', # Make sure these images exist!
    # '/Users/emadhekar/Pictures/alaska_cooking.JPG',
    # '/Users/emadhekar/Pictures/ganapti_challe_gavala.jpg',
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
                'role': 'system',
                'content': 'A chat between a curious human and an artificial intelligence assistant. The assistant is an expert in people, '
                'emotions and locations, and gives thoughtful, helpful, detailed, and polite answers to the human questions. '
                'Do not hallucinate and gives very close attention to the details and takes time to process information provided, '
                'response must be entirely in prose, absolutely no lists, bullet points, or numbered items should be used. Ensure the information flows seamlessly within paragraphs.'
                'Adhere strictly to these guidelines:'
                '1. Only provide answer and no extra commentary, additional context or information request.'
                '2. Do not reuse the same sentence structure more than once in response.'
                '3. Eliminate unclear excessive symbols or gibberish.'
                '4. Include addition information provided about people names and places or locations.'
                '5. Shorten text while preserving information.'
                '6. Preserve clear text as is.'
                '7. Skip text that is too unclear or ambiguous.'
                '8. Exclude non-factual elements.'
                '9. Maintain clarity and information.',
            },
                {
                    'role': 'user',
                    'content': 'Describe this image in detail.',
                    'images': [image_path] # LLaVA uses image paths in messages
                }
            ],
            options={'num_predict': 300,'temperature': 0.7 }, # Control response length
            stream=False # Get full response at once
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
    #server_urls = [f"{OLLAMA_HOST_BASE}{OLLAMA_PORT_START + i}" for i in range(2)]#range(len(image_paths))]
    server_urls = ["http://localhost:11434", "http://localhost:11435","http://localhost:11434", "http://localhost:11435", "http://localhost:11434"]
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
