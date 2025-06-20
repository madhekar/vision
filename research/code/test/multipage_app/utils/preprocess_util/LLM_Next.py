from transformers import pipeline
from PIL import Image

def init():

    model_id = "xtuner/llava-phi-3-mini-hf"
    pipe = pipeline("image-to-text", model=model_id, device="cpu")
    return pipe

"""


[{'generated_text': '<|im_start|>system\n    A chat between a curious human and an artificial intelligence assistant. 
The assistant is an expert in people, emotions and locations, and gives thoughtful, helpful, detailed, and polite answers to the human questions. \n    
The assistant does not hallucinate and gives extreamly very close attention to the details and take time to process information if necessary, please produce response in string format.\n    
<|im_end|>\n    
<|im_start|>user\n     "Please take time to describe the picture with thoughtful insights" It is stremly important that, you "MUST" include person names "Esha, Shibangi and 1 person" with "happy" emotion and the location details "Poway Performing Arts Theater" in the response.\n    
<|im_end|> \n    
<|im_start|>assistant\n    
1. Esha, a young woman, is sitting on the floor in the center of the stage, wearing a vibrant pink and green sari. She is holding a fan in her hand and has a radiant smile on her face.\n    
2. Shibangi, another young woman, is standing to the left of Esha. She is wearing a blue and gold sari and is holding a fan in her hand. She is looking directly at the camera with a serious expression.\n    
3. The third person, whose face is not visible in the image, is standing to the right of Esha. She is wearing a pink and green sari and is holding a fan in her hand. She is looking towards the left side of the stage.\n    
4. The stage is set against a purple background, which contrasts with the colorful saris of the performers.\n    
5. The Poway Performing Arts Theater is the location of this performance. It is a large, open space with a stage at the front and rows of seating on either side.'}]


[{'generated_text': 'system\n    A chat between a curious human and an artificial intelligence assistant. 
The assistant is an expert in people, emotions and locations, and gives thoughtful, helpful, detailed, and polite answers to the human questions. \n   
The assistant does not hallucinate and gives extreamly very close attention to the details and take time to process information if necessary, 
please produce response in string format.\n    \n    user\n     "Please take time to describe the picture with thoughtful insights" It is stremly important that, you "MUST" include person names "Esha, Shibangi and 1 person" with "happy" emotion and the location details "Poway Performing Arts Theater" in the response.\n     \n    
assistant\n    
1. Esha, a young woman, is sitting on the floor in the center of the stage, wearing a vibrant pink and green sari. She is holding a fan in her hand and has a radiant smile on her face.\n    
2. Shibangi, another young woman, is standing to the left of Esha. She is wearing a blue and gold sari and is holding a fan in her hand. She is looking directly at the camera with a serious expression.\n    
3. The third person, whose face is not visible in the image, is standing to the right of Esha. She is wearing a pink and green sari and is holding a fan in her hand. She is looking towards the left side of the stage.\n    
4. The stage is set against a purple background, which contrasts with the colorful saris of the performers.\n    
5. The Poway Performing Arts Theater is the location of this performance. It is a large, open space with a stage at the front and rows of seating on either side.'}]


/home/madhekar/work/home-media-app/data/input-data/img/Grad Pics family1.jpg
    2 women are hugging in front of the UC Berkeley Sather Gate. 
    One of them is wearing a graduation cap and gown. The other woman is wearing a gray sweater. 
    There are 4 other people in the background, one of whom is holding a camera. The sky is blue and there are trees in the background. 
    The women are smiling and appear to be happy. The graduation cap and gown suggest that one of the women has recently graduated from college. 
    The camera in the hands of one of the people in the background suggests that they are taking photos to commemorate the occasion. 
    The blue sky and trees in the background give the image a pleasant and serene atmosphere.

/home/madhekar/work/home-media-app/data/input-data/img/20130324-3I3A4652-X2.jpg
    3 women are performing a traditional Indian dance on a stage at the Poway Performing Arts Theater. 
    The stage is set against a purple background, adding a vibrant touch to the performance.
    \n\nThe woman on the left is kneeling on the ground, her hands raised in a gesture of joy and celebration. 
    She is wearing a traditional Indian outfit in shades of pink and green, adorned with gold jewelry that glitters under the stage lights.
    \n\nIn the center, another woman is standing, her hands gracefully extended as if she is leading the dance. 
    She is dressed in a similar traditional Indian outfit, but her colors are more muted, with hues of blue and white.\n\nOn the right, the third woman is also standing, her hands held close to her body. 
    She is wearing a traditional Indian outfit in shades of pink and blue, with gold jewelry that complements her attire.
    \n\nAll three women are smiling, their faces radiating happiness and enthusiasm. 
    Their traditional Indian outfits and jewelry add a touch of cultural richness to the performance, making it a truly captivating sight to behold.


    3 women are performing a traditional Indian dance on a stage at the Poway Performing Arts Theater. 
    The stage is set against a purple background, creating a vibrant and colorful atmosphere.
    \n\nThe woman on the left is sitting on the floor, adorned in a traditional Indian outfit that is predominantly pink and green. 
    She holds a fan in her hand, adding to the authenticity of her performance.\n\nIn the center, another woman is standing, wearing a similar outfit but with a more balanced mix of pink and blue colors. 
    She too holds a fan, mirroring the actions of the woman on the left.
    \n\nOn the right, the third woman is standing and wearing an outfit that is a mix of pink and green. 
    She holds a flower in her hand, adding a touch of elegance to her performance.
    \n\nAll three women are smiling, their faces radiating happiness and joy. 
    Their traditional Indian outfits, complete with jewelry and accessories, add a cultural richness to the scene. 
    The Poway Performing Arts Theater serves as the perfect backdrop for this captivating performance.

    1. Esha, a young woman, is sitting on the floor in the center of the stage, wearing a vibrant pink and green sari. She is holding a fan in her hand and has a radiant smile on her face.\n    
    2. Shibangi, another young woman, is standing to the left of Esha. She is wearing a blue and gold sari and is holding a fan in her hand. She is looking directly at the camera with a serious expression.\n    
    3. The third person, whose face is not visible in the image, is standing to the right of Esha. She is wearing a pink and green sari and is holding a fan in her hand. She is looking towards the left side of the stage.\n    
    4. The stage is set against a purple background, which contrasts with the colorful saris of the performers.\n    
    5. The Poway Performing Arts Theater is the location of this performance. It is a large, open space with a stage at the front and rows of seating on either side.
"""

def fetch_llm_text(imUrl, pipe, question, people, attrib, location):

    image = Image.open(imUrl)
    
    prompt = """<|im_start|>system
    A chat between a curious human and an artificial intelligence assistant. The assistant is an expert in people, emotions and locations, and gives thoughtful, helpful, detailed, and polite answers to the human questions. 
    The assistant does not hallucinate and gives extreamly very close attention to the details and take time to process information if necessary, please produce response in string format.
    <|im_end|>
    <|im_start|>user
    <image>"{question}" It is stremly important that, you "MUST" include person names "{people}" with "{attrib}" emotion and the location details "{location}" in the response.
    <|im_end|> 
    <|im_start|>assistant
    """.format(question=question, people=people, attrib=attrib, location=location) #, article=st.session_state["document"])

    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 400})
    return outputs, prompt
    
# make a call    
#url = "/home/madhekar/work/home-media-app/data/input-data/img/Grad Pics family1.jpg"
url= '/home/madhekar/work/home-media-app/data/input-data/img/20130324-3I3A4652-X2.jpg'
p = init()
result, prompt = fetch_llm_text(url, p, "Please take time to describe the picture with thoughtful insights", "Esha, Shibangi and 1 person", "happy", "Poway Performing Arts Theater" )
print(result)
print('\n\n')
res = str(result)
r = res.replace(prompt, "").replace("<|im_end|>", "").replace("<|im_start|>", "")

print(r)