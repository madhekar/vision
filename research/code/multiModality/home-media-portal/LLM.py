
import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from transformers import TextStreamer

"""
https://github.com/haotian-liu/LLaVA/issues/269

Automodel is a term used to describe a machine learning model that can be automatically generated or trained. 
This can be done in a variety of ways, such as using AutoML tools or by using techniques such as transfer learning.
Automodels are often used in machine learning because they can significantly reduce the time and effort required to develop and deploy models. 
For example, an AutoML tool can be used to automatically generate a variety of different model architectures and then train and evaluate them on a given dataset. 
This can help users to quickly identify the best model architecture for their specific needs.
Automodels are also becoming increasingly popular in the field of natural language processing (NLP). 
This is because AutoNLP tools can be used to automatically generate and train language models for a variety of tasks, such as text classification, question answering, and machine translation.

Have you ever wondered how advanced AI models can interpret images and respond to queries about them? Introducing Multi-Crop LLaVA-3b, 
an innovative model that allows AI to extract visual information from various parts of an image by creating multiple tokens for those segments. 
This tutorial will walk you through using the Multi-Crop LLaVA-3b model with some user-friendly steps.

Imagine you have a garden with a variety of flowers, and you want to write a poem about each flower individually rather than just a single poem about the entire garden. 
Instead of creating all your sentiments based on just one perspective of the garden, you observe multiple angles and specific flowers, gathering detailed information that enriches your final piece. 
This is exactly how the Multi-Crop LLaVA model works by dissecting images into several smaller parts (or crops), analyzing them individually, and combining the insights to generate accurate responses.

from transformers import AutoModel, AutoProcessor
import torch

Load the Model: Initialize the model and processor.

model = AutoModel.from_pretrained("visheratin/MC-LLaVA-3b", torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
processor = AutoProcessor.from_pretrained("visheratin/MC-LLaVA-3b", trust_remote_code=True)

Process Inputs: Prepare your inputs with the image and prompt text.

with torch.inference_mode():
    inputs = processor(prompt, [raw_image], model, max_crops=100, num_tokens=728)
    output = model.generate(**inputs, max_new_tokens=200, use_cache=True, do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.eos_token_id)

Decode and Print Result: Retrieve and display the model’s output.

result = processor.tokenizer.decode(output[0]).replace(prompt, "").replace(im_end, "")
print(result)
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
def fetch_llm_text(imUrl, model, processor, top, temperature, question, people, location):
    
    prompt = """<|im_start|>system
    A chat between a curious human and an artificial intelligence assistant. The assistant is an expert in people and locations, and gives helpful, detailed, and polite answers to the human's questions. The assistant does not hallucinate and pays very close attention to the details.
    <|im_end|>
    <|im_start|>user
    <image>
     {question}you must include person name(s) "{people}" and the location details "{location}" in the answer.
    <|im_end|> 
    <|im_start|>assistant
    """.format(question=question, people=people, location=location) #, article=st.session_state["document"])

    # generate propcssor using image and associated prompt query, and generate LLM response
    with torch.inference_mode():
        inputs = processor(
            prompt, [Image.open(imUrl)], model, max_crops=100, num_tokens=728        
            )
        
    # streamer 
    streamer = TextStreamer(processor.tokenizer)

    # model generator
    with torch.inference_mode():
        output =  model.generate(
            **inputs,
            max_new_tokens=200,
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
    result = result.replace(prompt, "").replace("<|im_end|>", "").replace("<|im_start|>", "")
    return result
    
