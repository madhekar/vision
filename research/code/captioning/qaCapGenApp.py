import streamlit as st
from streamlit_image_select import image_select
import torch
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

st.set_page_config(
    page_title="zesha: Multi Modality Search (MMS)",
    page_icon="",
    initial_sidebar_state="auto",
    layout="wide",
)  # (margins_css)

iroot = "/home/madhekar/work/zsource/family/img/"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("microsoft/git-base")


def loadModel():
    return torch.load("./zgit")

def generate(image):    
    # prepare image for the model
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values

    model = loadModel()
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return generated_caption

if __name__ == "__main__":
    st.title('GIT image caption Model test')
    
    selected_files = st.sidebar.file_uploader('select image files', accept_multiple_files=True)
    print(selected_files)  
    
    gen_cap = []
    imfiles = []
    if selected_files:
        for f in selected_files:
            print(iroot + f.name)
            gen_cap.append(generate(Image.open(iroot +f.name)))
            imfiles.append(iroot+ f.name)
            
    btn = st.sidebar.button('generate captions')        
    if btn:    
       dimgs = image_select(
          label="Select Image",
          images=imfiles,
          use_container_width=True,
          captions=gen_cap
       )