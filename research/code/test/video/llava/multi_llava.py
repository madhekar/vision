import os
import ollama
import base64
import asyncio


"""
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
Environment="OLLAMA_KEEP_ALIVE=10m"
Environment="OLLAMA_DEBUG=1"
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
"""
# Function to encode the image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def create_default_client():
    client = ollama.Client()
    return client

def caption_image(img):

    client = ollama.Client()
    if not os.path.exists(img):
        print(f"Err: Image path does not exists {img}")
        return ""
    else:    

       result = client.chat(
            model= 'mapler/llama3-llava-next-8b:latest', #'llava',
            messages=[  
                {
                'role': 'user',
                'content': 'Act as professional copywriter. Write an engaging caption for this image.',
                'images': [img]
                }
            ]
       )
       return result["message"]["content"]   

def describe_image( frames_path, ppt, location):

    client = ollama.Client()
    
    eimg_list = []
    print(f"{frames_path}")
    for img in os.listdir(frames_path):
       print(img)
       eimg_list.append(encode_image_to_base64(os.path.join(frames_path,img)))
    

    # prompt
    if location != "" and ppt != "":
        prompt = f"Describe the video frames with thoughtful insights by connecting them in single coherant response. you must include names of people {ppt} and location {location} in response"
    elif location == "" and ppt != "":
        prompt = f"Describe the video frames with thoughtful insights by connecting them in single coherant response. you must include names of people {ppt} in response"
    elif location != "" and ppt == "":
        prompt = f"Describe the video frames with thoughtful insights by connecting them in single coherant response. you must include location {location} in response"
    else: 
       prompt = "Describe the video frames with thoughtful insights  by connecting them in single coherant response."
    try:
        # Perform inference
        response = client.chat( 
            model= 'mapler/llama3-llava-next-8b:latest', #'llava',
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
                    'images': eimg_list,
                }
            ],
            options={'num_predict': 500,  'temperature': 0.8},
            stream=False # Get full response at once
        )

        return response['message']['content']
    except Exception as e:
        print(f"Error processing image...: {e}")
        return None
    # finally:
    #     if client:
    #        client.close() # Important to close the client

if __name__ == "__main__":
    # Path to your image
    frames_path =  "/home/madhekar/Videos/ffmpeg_frames/video_1/frames"
    location = "Indor, MP, India"
    ppt = "Asha"
    
    client = create_default_client()

    m = describe_image(
        frames_path,
        ppt,
        location
        )   
     
    print(f"describe video {m}")

    # image_for_caption = "/mnt/zmdata/home-media-app/data/final-data/img/Berkeley/80f3b1f6-baca-5e05-9cf8-2046abda2260/IMG_5380.jpeg"

    # cap = caption_image(image_for_caption)

    # print(f"caption : {cap}")