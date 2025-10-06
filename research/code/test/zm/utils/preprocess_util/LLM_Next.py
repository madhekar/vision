from transformers import pipeline
from PIL import Image
import streamlit as st
import re

"""
from transformers import pipeline
from PIL import Image    


model_id = "xtuner/llava-llama-3-8b-v1_1-transformers"
pipe = pipeline("image-to-text", model=model_id, device="cpu")
img = "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/IMG_5156.PNG"
#"/home/madhekar/work/home-media-app/data/input-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/IMG_5466.PNG"#"http://images.cocodataset.org/val2017/000000039769.jpg"

image = Image.open(img)#requests.get(url, stream=True).raw)
prompt = (
    "<|start_header_id|>user<|end_header_id|>\n\n<image>\nPlease take time to describe the image with thoughtful insights<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)
outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)
[{'generated_text': 'user\n\n\nWhat are these?assistant\n\nThese are two cats, one brown and one gray, lying on a pink blanket. sleep. brown and gray cat sleeping on a pink blanket.'}]

"""
@st.cache_resource(ttl=36000, show_spinner=True)
def setLLM():

    model_id = "xtuner/llava-llama-3-8b-v1_1-transformers" #"xtuner/llava-phi-3-mini-hf" #"xtuner/llava-llama-3-8b-hf"
    pipe = pipeline("image-to-text", model=model_id, device="cpu")
    return pipe

def fetch_llm_text(imUrl, pipe, question, partial_prompt, location):

    st.info("calling LLM...")

    image = Image.open(imUrl).convert("RGB")
    # Ensure that all double and single quotes are escaped with backslash: "This is a \"quoted\" pharse." or 'I\'m going now'.
    if partial_prompt != '':
    
        prompt = """<|im_start|>system
        A chat between a curious human and an artificial intelligence assistant. The assistant is an expert in people, emotions and locations, and gives thoughtful, helpful, detailed, and polite answers to the human questions. 
        Do not hallucinate and gives very close attention to the details and takes time to process information provided, your response must be entirely in prose. Absolutely no lists, bullet points, or numbered items should be used. 
        Ensure the information flows seamlessly within paragraphs. Adhere strictly to these guidelines:
        1. Only provide answer and no extra commentry, additional context or information request.
        2. Eliminate unclear text, such as excessive symbols or gibberish.
        3. Shorten text while preserving information.
        4. Preserve clear text as is.
        5. Skip text that is too unclear or ambiguous.
        6. Exclude non-factual elements.
        7. Maintain clearity and information.
        <|im_end|>
        <|im_start|>user
        <image>"{question}" It is CRITICALLY important to include the NAMES OF PEOPLE and EMOTIONS if provided "{partial_prompt}" and the location details "{location}" in the response if appropriate.  
        <|im_end|> 
        <|im_start|>assistant
        """.format(
            question=question, partial_prompt=partial_prompt, location=location
        )  # , article=st.session_state["document"])
    else:
        prompt = """<|im_start|>system
        A chat between a curious human and an artificial intelligence assistant. The assistant is an expert in people, emotions and locations, and gives thoughtful, helpful, detailed, and polite answers to the human questions. 
        Do not hallucinate and gives very close attention to the details and takes time to process information provided, your response must be entirely in prose. Absolutely no lists, bullet points, or numbered items should be used. 
        Ensure the information flows seamlessly within paragraphs.
        <|im_end|>
        <|im_start|>user
        <image>"{question}" please use the location details "{location}" in the response if appropriate.
        <|im_end|> 
        <|im_start|>assistant
        """.format(
            question=question, location=location
        )  # , article=st.session_state["document"])

    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

    result = outputs[0]["generated_text"].partition("<|im_start|>assistant")[2]
    #rr = repr(result)
    rr = result.translate(str.maketrans({
                                        #   "-":  r"\-",
                                        #   "]":  r"\]",
                                        #   "\\": r"\\",
                                        #   "^":  r"\^",
                                        #   "$":  r"\$",
                                        #   "*":  r"\*",
                                        #   ".":  r"\.",
                                          "'" :  r"\'",
                                          "|":  r""
                                        #  '"':  r"\""
                                          }))
    return rr
    
if __name__=='__main__':
    url= '/home/madhekar/work/home-media-app/data/input-data/img/20130324-3I3A4652-X2.jpg'
    p = setLLM()
    generation_args = {"max_new_tokens": 200,"return_full_text": False,"temperature": 0.0,"do_sample": False,}
    result = fetch_llm_text(url, p, "Please take time to describe the picture with thoughtful insights", "Esha, Shibangi and 1 person", "happy", "Poway Performing Arts Theater" )
    print(result)