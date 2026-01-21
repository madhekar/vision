import ollama
import base64


'''
Now we need to enable vulkan, allow the Ollama server to talk to your local network, and allow flash attention. Flash attention is optional, but recommended. In a terminal enter the following:

sudo systemctl edit ollama
You’ll get a nano editor with the service’s settings. Add the following lines after the “### Anything between here and the comment below<…>” to set environmental variables.

The OLLAMA_HOST line will allow anything on your local network (the 0.0.0.0 part) to find and talk to the server. If you are working purely local to your machine, don’t add this.
The OLLAMA_VULKAN line requests Ollama to use vulkan rather than rocm which has been problematic.
The OLLAMA_FLASH_ATTENTION line will enable flash attention which is ideal for larger contexts. The last line is optional and will depend on your needs.
The OLLAMA_CONTEXT_LENGTH line allows you to set the context window size.
Note that not all models will support large contexts. Change this value up or down to suit your needs. Be aware that large contexts may also degrade performance.
The default if you don’t add this line is 4096 tokens which is not many. 32k is a good number for general coding. I typically set mine to 50k or more if the model can handle it.
### Editing /etc/systemd/system/ollama.service.d/override.conf

### Anything between here and the comment below will become the contents of the drop-in file

[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_VULKAN=1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_CONTEXT_LENGTH=32768"
'''
# Function to encode the image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def create_default_client():
    client = ollama.Client(host="http://localhost:11434")
    return client

def describe_image(client, img_path, ppt, location):

    encoded_image = encode_image_to_base64(img_path)

    # prompt
    if location != "" and ppt != "":
        prompt = f"Describe the image with thoughtful insights using information provided. you must include names of people {ppt} and location {location} in response"
    elif location == "" and ppt != "":
        prompt = f"Describe the image with thoughtful insights using information provided. you must include names of people {ppt} in response"
    elif location != "" and ppt == "":
        prompt = f"Describe the image with thoughtful insights using information provided. you must include location {location} in response"
    else: 
       prompt = f"Describe the image with thoughtful insights in response."

    # Perform inference
    response = client.chat( #ollama.chat(
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
                'images': [encoded_image]
            }
        ],
        options={'temperature': 0.7},
        stream=False # Get full response at once
    )

    return response['message']['content']

if __name__ == "__main__":
    # Path to your image
    image_path = '/home/madhekar/temp/filter/training/people/IMG_1531.jpeg'
    location = "Madhekar residence in San Diego"
    ppt = "Anjali and Esha"
    
    client = create_default_client()

    m =  describe_image(client,
        image_path, 
        ppt,
        location
        )   

    print(m)