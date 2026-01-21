import multiprocessing
import ollama
import time
from functools import partial

# Define the function for Ollama inference
def run_inference(prompt):
    # This client is created within the process, connecting to the default local Ollama server
    client = ollama.Client(host='http://localhost:11434') 
    try:
        start_time = time.time()
        response = client.generate(
            model='mistral',  # Use the model you have pulled
            prompt=prompt
        )
        end_time = time.time()
        return f"Prompt: {prompt[:30]}... | Time: {end_time - start_time:.2f}s | Response snippet: {response['response'][:50]}..."
    except Exception as e:
        return f"Error with prompt '{prompt[:30]}...': {e}"

if __name__ == '__main__':
    # Ensure the code runs only when executed as a script
    
    # List of prompts to process in parallel
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a short story about a brave knight and a dragon.",
        "Provide a Python code snippet for a quick sort algorithm.",
        "What are the benefits of regular exercise?",
        "Describe the plot of the movie Inception.",
        "How does a combustion engine work?",
        "What is the capital of Australia?",
        "Explain the concept of 'black holes'."
    ]

    # Number of parallel processes (adjust based on your system resources and Ollama's capacity)
    num_processes = 4 

    print(f"Starting parallel inference for {len(prompts)} prompts with {num_processes} processes...")
    start_total_time = time.time()

    # Use multiprocessing Pool to distribute the tasks
    with multiprocessing.Pool(processes=num_processes) as pool:
        # map() distributes the `run_inference` function across the list of prompts
        results = pool.map(run_inference, prompts)

    end_total_time = time.time()
    print("\n--- Results ---")
    for result in results:
        print(result)

    print(f"\nTotal time taken: {end_total_time - start_total_time:.2f}s")
