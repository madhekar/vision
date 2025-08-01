
import streamlit as st
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from transformers import TextStreamer

"""
Automodel is a term used to describe a machine learning model that can be automatically generated or trained. 
This can be done in a variety of ways, such as using AutoML tools or by using techniques such as transfer learning.
Automodels are often used in machine learning because they can significantly reduce the time and effort required to develop and deploy models. 
For example, an AutoML tool can be used to automatically generate a variety of different model architectures and then train and evaluate them on a given dataset. 
This can help users to quickly identify the best model architecture for their specific needs.
Automodels are also becoming increasingly popular in the field of natural language processing (NLP). 
This is because AutoNLP tools can be used to automatically generate and train language models for a variety of tasks, such as text classification, question answering, and machine translation.
"""
@st.cache_resource(ttl=36000, show_spinner=True)
def setLLM():
    """
    model auto-tokenizer and processor components for LLM model MC-LLaVA-3b with trust flag
    """

    model = AutoModel.from_pretrained("visheratin/MC-LLaVA-3b", trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "visheratin/MC-LLaVA-3b", trust_remote_code=True, batched=True
    )

    processor = AutoProcessor.from_pretrained(
        "visheratin/MC-LLaVA-3b", trust_remote_code=True
    )

    return model, tokenizer, processor

    # create prompt to test the LLM
    # Do not write outside its scope unless you find your answer better {article} if you thin your answer is better add it after document.<|im_end|>
def fetch_llm_text(imUrl, model, processor, top, temperature, question, people, attrib, location):
    
    st.info("calling LLM...")

    prompt = """<|im_start|>system
    A chat between a curious human and an artificial intelligence assistant. The assistant is an expert in people, emotions and locations, and gives thoughtful, helpful, detailed, and polite answers to the human questions. 
    The assistant does not hallucinate and gives very close attention to the details and take time to process information if necessary.
    <|im_end|>
    <|im_start|>user
    <image>
     "{question}" you MUST include person name(s) "{people}" with "{attrib}" emotion and the location details "{location}" in the answer.
    <|im_end|> 
    <|im_start|>assistant
    """.format(question=question, people=people, attrib=attrib, location=location) #, article=st.session_state["document"])
    image = Image.open(imUrl).convert('RGB')
    #image = np.array(image)
    #image = image[:, :, :3]
    # generate propcssor using image and associated prompt query, and generate LLM response
    with torch.inference_mode():
        inputs = processor( prompt, [image], model, max_crops=100, num_tokens=200)        #100, 72 (100)
        
    # streamer 
    streamer = TextStreamer(processor.tokenizer)

    # model generator
    with torch.inference_mode():
        output =  model.generate(
            **inputs,
            max_new_tokens=200,  #200 #70
            do_sample=True,
            use_cache=False,
            top_p=top,
            temperature=temperature,
            eos_token_id=processor.tokenizer.eos_token_id,
            streamer=streamer,
            #return_dict_in_generate=True,
            #output_scores=True
        )

    result = processor.tokenizer.decode(output[0])
    r = result.replace(prompt, "").replace("<|im_end|>", "").replace("<|im_start|>", "")

    return r
    
