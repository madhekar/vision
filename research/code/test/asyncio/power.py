# main.py
import asyncio
import json

async def run_power_subprocess():
    # Define the inputs to pass to the script
    base = 2
    exponents = [1, 2, 3, 4, 5]
    
    # Serialize the collection argument to a JSON string
    exponents_json = json.dumps(exponents)
    
    # Start the subprocess
    process = await asyncio.create_subprocess_exec(
        'python3', 'worker.py', str(base), exponents_json,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Wait for the process to finish and capture outputs
    stdout, stderr = await process.communicate()
    
    if process.returncode == 0:
        # Decode bytes to string and parse JSON back into a Python collection
        result_collection = json.loads(stdout.decode().strip())
        print(f"Returned Collection: {result_collection}")
        print(f"Type: {type(result_collection)}")
    else:
        print(f"Error occurred: {stderr.decode()}")

# Run the async loop
if __name__ == "__main__":
    asyncio.run(run_power_subprocess())
