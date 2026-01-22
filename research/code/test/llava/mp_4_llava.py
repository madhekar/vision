import ollama
import base64
from PIL import Image
import io
from concurrent.futures import ProcessPoolExecutor

# Function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to query Ollama (designed to run in a separate process)
def query_ollama_llava(task_data):
    # This function runs in its own process, so it needs to import ollama
    import ollama
    prompt = task_data['prompt']
    image_path = task_data['image_path']
    try:
        # Encode the image within the process to avoid sharing large objects between processes
        b64_image = encode_image_to_base64(image_path)
        
        # Call the Ollama API with multimodal input
        response = ollama.chat(
            model='llava',
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
                    'content': prompt,
                    'images': [b64_image] # Pass the base64 encoded image
                }
            ]
        )
        return f"Task for '{image_path}': {response['message']['content']}"
    except Exception as e:
        return f"Error processing task for '{image_path}': {e}"

if __name__ == '__main__':
    # Create some dummy image files for the example
    # In a real scenario, you would use your own images
    img1_path = "image1.jpg"
    img2_path = "image2.jpg"
    Image.new('RGB', (100, 100), color = 'red').save(img1_path)
    Image.new('RGB', (100, 100), color = 'blue').save(img2_path)

    # Define the tasks
    tasks = [
        {'prompt': 'What color is the image?', 'image_path': img1_path},
        {'prompt': 'Describe the background color.', 'image_path': img2_path},
        {'prompt': 'Identify the primary color.', 'image_path': img1_path},
        {'prompt': 'What color is this?', 'image_path': img2_path},
    ]

    print(f"Starting {len(tasks)} Ollama LLaVA tasks using multiprocessing...")

    # Use ProcessPoolExecutor to run tasks in parallel
    # Max workers can be adjusted based on available CPU cores and GPU memory
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Map the function to the list of tasks
        results = executor.map(query_ollama_llava, tasks)

    print("\nResults:")
    for result in results:
        print(result)

    # Clean up dummy files
    import os
    os.remove(img1_path)
    os.remove(img2_path)

'''
Results:
Task for 'image1.jpg':  The image is red. 
Task for 'image2.jpg':  The background color is a deep blue. 
Task for 'image1.jpg':  The primary color in the image you've provided is red. There are no other dominant colors that take precedence over this one. Red is often associated with passion, energy, and sometimes danger or urgency, depending on its shade and context. 
Task for 'image2.jpg':  This image features a solid blue color. 
'''