import asyncio
from ollama import AsyncClient

async def process_image(model, prompt, image_path):
    client = AsyncClient()
    try:
        response = await client.chat(model=model, messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_path]
        }])
        return response['message']['content']
    except Exception as e:
        return f"Error processing {image_path}: {e}"

async def main():
    model = 'llava'
    prompt = 'Describe this image in detail'
    # Example images - replace with your image paths
    images = [
            '/home/madhekar/temp/filter/training/people/IMG_5543.jpeg',
            '/home/madhekar/temp/filter/training/people/neighbors.jpg',
            '/home/madhekar/temp/filter/training/people/IMG_5533.jpeg',
            '/home/madhekar/temp/filter/training/people/IMG_7285.jpeg',
            '/home/madhekar/temp/filter/training/people/IMG_7460.jpeg'
              ]
    
    # Using asyncio.gather for concurrent execution (acting as a "pool" of tasks)
    tasks = [process_image(model, prompt, img) for img in images]
    results = await asyncio.gather(*tasks)
    
    for i, res in enumerate(results):
        print(f"Result {i+1}: {res}\n---")

if __name__ == '__main__':
    asyncio.run(main())
