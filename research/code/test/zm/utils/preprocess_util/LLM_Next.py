from transformers import pipeline
from PIL import Image
import streamlit as st

@st.cache_resource(ttl=36000, show_spinner=True)
def setLLM():

    model_id = "xtuner/llava-phi-3-mini-hf"
    pipe = pipeline("image-to-text", model=model_id, device="cpu")
    return pipe

def fetch_llm_text(imUrl, pipe, question, partial_prompt, location):

    st.info("calling LLM...")

    image = Image.open(imUrl).convert("RGB")
    
    prompt = """<|im_start|>system
    A chat between a curious human and an artificial intelligence assistant. The assistant is an expert in people, emotions and locations, and gives thoughtful, helpful, detailed, and polite answers to the human questions. 
    The assistant does not hallucinate and gives extreamly very close attention to the details and take time to process information if necessary, please produce plain text response "avoiding emojis or lists" in response.
    <|im_end|>
    <|im_start|>user
    <image>"{question}" It is extremely important that, you "MUST" include the people details "{partial_prompt}" and the location details "{location}" in the response.
    <|im_end|> 
    <|im_start|>assistant
    """.format(question=question, partial_prompt=partial_prompt, location=location)  # , article=st.session_state["document"])
 
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

    result = outputs[0]["generated_text"].partition("<|im_start|>assistant")[2]

    return result
    
if __name__=='__main__':
    url= '/home/madhekar/work/home-media-app/data/input-data/img/20130324-3I3A4652-X2.jpg'
    p = setLLM()
    generation_args = {"max_new_tokens": 200,"return_full_text": False,"temperature": 0.0,"do_sample": False,}
    result = fetch_llm_text(url, p, "Please take time to describe the picture with thoughtful insights", "Esha, Shibangi and 1 person", "happy", "Poway Performing Arts Theater" )
    print(result)