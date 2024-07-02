import streamlit as st
import torch

# from streamlit_option_menu import option_menu
from streamlit_image_select import image_select
from PIL import Image

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from transformers import AutoTokenizer, AutoProcessor, AutoModel, TextStreamer
import chromadb.utils.embedding_functions
from loadData import init, setLLM
# import chromadb as cdb

st.set_page_config(
    page_title="zesha flower search",
    page_icon="",
    initial_sidebar_state="auto",
    layout="wide",
)  # (margins_css)
st.title("Flower Search")

# load data
cImgs, cTxts = init()

# init LLM modules
m, t, p = setLLM()


st.sidebar.title("seach criteria")
# with st.sidebar.form(key='attributes'):

ms = st.sidebar.multiselect(
    "select search result modalities", ["img", "txt", "video", "audio"]
)

sim = st.sidebar.file_uploader("search doc/image: ", type=["png", "jpeg", "mpg"])

im = None
if sim:
    im = Image.open(sim)
    name = sim.name
    st.sidebar.image(im, caption="select image")
    with open(name, "wb") as f:
        f.write(sim.getbuffer())


txt = st.sidebar.text_input(
    "enter the search term: ", placeholder="enter short search term", disabled=False
)

top = st.sidebar.slider("select top results pct", 0.0, 1.0, (0.0, 1.0))

te = st.sidebar.slider("select LLM temperature: ", 0.0, 2.0, (0.0, 1.0))

btn = st.sidebar.button(label="Search")

if btn:
    
    # create query on image, also shows similar document in vector database (not using LLM)
    # openclip embedding function!
    embedding_function = OpenCLIPEmbeddingFunction()

    question = "Answer with organized answers: What type of flower is in the picture? Mention some of its characteristics and how to take care of it ?"

    st.session_state["document"] = cTxts.query(
        query_embeddings=embedding_function("./" + sim.name),
        n_results=1,
    )["documents"][0][0]

    if "timgs" in st.session_state:
        st.session_state["timgs"] = []

    imgs = cImgs.query(query_uris="./" + sim.name, include=["data"], n_results=6)
    # st.write('=>images: ', imgs)
    for img in imgs["data"][0][1:]:
        st.session_state["timgs"].append(img)

    
    # create prompt to test the LLM
    
    prompt = """<|im_start|>system
    A chat between a curious human and an artificial intelligence assistant.
    The assistant is an exprt in flowers , and gives helpful, detailed, and polite answers to the human's questions.
    The assistant does not hallucinate and pays very close attention to the details.<|im_end|>
    <|im_start|>user
    <image>
    {question} Use the following article as an answer source. Do not write outside its scope unless you find your answer better {article} if you thin your answer is better add it after document.<|im_end|>
    <|im_start|>assistant
    """.format(question="question", article=st.session_state["document"])

    
    # generate propcssor using image and associated prompt query, and generate LLM response
    
    with torch.inference_mode():
        inputs = p(prompt, [Image.open(sim)], m, max_crops=100, num_tokens=728)

    streamer = TextStreamer(p.tokenizer)

    with torch.inference_mode():
      output = m.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        use_cache=False,
        top_p=0.9,
        temperature=0.2,
        eos_token_id=p.tokenizer.eos_token_id,
        streamer=streamer,
       )

    st.text_area(
       label="LLM",
        value=p.tokenizer.decode(output[0]).replace(prompt, "").replace("<|im_end|>", ""),
    )


if "document" not in st.session_state:
    st.session_state["document"] = ""

if "timgs" not in st.session_state:
    st.session_state["timgs"] = []

if len(st.session_state["timgs"]) > 1:
    st.text_area(label="flower description", value=st.session_state["document"])

    dimgs = image_select(
        label="select image",
        images=st.session_state["timgs"],
        use_container_width=True,
        captions=[
            "caption one",
            "caption two",
            "caption three",
            "caption fore",
            "caption five",
        ],
    )

    st.image(dimgs, use_column_width="always")
