
import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from transformers import TextStreamer


@st.cache_resource(ttl=36000, show_spinner=True)
def setLLM():
    """
    model autotokenizer and processor componnents for LLM model MC-LLaVA-3b with trust flag
    """

    model = AutoModel.from_pretrained("visheratin/MC-LLaVA-3b", trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "visheratin/MC-LLaVA-3b", trust_remote_code=True, batched=True
    )

    processor = AutoProcessor.from_pretrained(
        "visheratin/MC-LLaVA-3b", trust_remote_code=True
    )

    return model, tokenizer, processor


def fetch_llm_text(imUrl, model, processor, top, temperature, question):
    
    # create prompt to test the LLM
    # Do not write outside its scope unless you find your answer better {article} if you thin your answer is better add it after document.<|im_end|>
    
    prompt = """
    <|im_start|>system
    A chat between a curious human and an artificial intelligence assistant.The assistant is an exprt in flowers , and gives helpful, detailed, and polite answers to the human's questions. The assistant does not hallucinate and pays very close attention to the details.
    <|im_end|>
    
    <|im_start|>user
    <image>
    {question} 
    <|im_end|> 
    
    <|im_start|>assistant
    """.format(question=question) #, article=st.session_state["document"])

    # generate propcssor using image and associated prompt query, and generate LLM response
    with torch.inference_mode():
        inputs = processor(
            prompt, [Image.open(imUrl)], model, max_crops=150, num_tokens=500
        )

    streamer = TextStreamer(processor.tokenizer)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            use_cache=False,
            top_p=top,
            temperature=temperature,
            eos_token_id=processor.tokenizer.eos_token_id,
            streamer=streamer,
        )
    #st.write('=>out: ', output )
    result = processor.tokenizer.decode(output[0])
    result = result.replace(prompt, "").replace("<|im_end|>", "").replace("<|im_start|>", "")
    return result
    
