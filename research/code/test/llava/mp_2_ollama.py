import multiprocessing
from ollama import Client
import time

def chat_session(host, model_name, user_prompt, system_prompt):
    """
    Handles a single chat session in a separate process.
    """
    # Create a client instance with the specific host/port
    client = Client(host=host)
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    
    print(f"--- Process for {model_name} starting ---")
    start_time = time.time()
    try:
        # The chat method is used to generate a single response here
        response = client.chat(model=model_name, messages=messages)
        end_time = time.time()
        print(f"--- Process for {model_name} finished in {end_time - start_time:.2f} seconds ---")
        print(f"Model: {model_name}, Response: {response['message']['content'][:100]}...")
    except Exception as e:
        print(f"Error in {model_name} process: {e}")

if __name__ == '__main__':
    # Ensure the code runs only when executed as a script
    # Define the tasks with their specific host/model
    tasks = [
        ('http://localhost:11434', 'llava:13b', 'What is the capital of France?', 'You are a helpful assistant.'),
        ('http://localhost:11435', 'llava', 'Explain the concept of black holes.', 'You are a concise science expert.'),
        ('http://localhost:11436', 'llava', 'Write a short poem about the ocean.', 'You are a creative poet.')
    ]

    processes = []
    # Create and start a process for each task
    for host, model, prompt, system_p in tasks:
        p = multiprocessing.Process(
            target=chat_session, 
            args=(host, model, prompt, system_p)
        )
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("\nAll chat processes have completed.")
