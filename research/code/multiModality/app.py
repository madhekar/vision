import streamlit as st
from PIL import Image
from streamlit_image_select import image_select
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from transformers import AutoTokenizer, AutoProcessor, AutoModel, TextStreamer
import chromadb.utils.embedding_functions
from loadData import init, setLLM
# import chromadb as cdb

st.set_page_config(page_title='zesha flower search', page_icon='', initial_sidebar_state='auto', layout='wide') #(margins_css)
st.title('Flower Search')

# load data
cImgs, cTxts = init()

# init LLM modules
m, t, p = setLLM()


st.sidebar.title('seach criteria')
with st.sidebar.form(key='attributes'):
  
     ms = st.multiselect('select search result modalities', ['img', 'txt', 'video', 'audio'])
  
     sim = st.file_uploader('search doc/image: ', type=['png', 'jpeg', 'mpg'])
  
     im = None
     if sim:
       im = Image.open(sim)
       name = sim.name
       st.image(im, caption="select image")
       with open(name, "wb") as f:
         f.write(sim.getbuffer())


     txt = st.text_input('enter the search term: ', placeholder='enter short search term', disabled=False)

     top = st.slider('select top results pct', 0.0,1.0, (0.0, 1.0))

     te = st.slider('select LLM temperature: ', 0.0, 2.0, (0.0, 1.0))

     btn = st.form_submit_button(label='process')

if btn:
  '''
     create query on image, also shows similar document in vector database (not using LLM)
  '''
  # openclip embedding function!
  embedding_function = OpenCLIPEmbeddingFunction()
  
  question = 'Answer with organized answers: What type of flower is in the picture? Mention some of its characteristics and how to take care of it ?'

  doc = cTxts.query(
    query_embeddings=embedding_function('./' + sim.name),
    n_results=1,
  )['documents'][0][0]

  st.text_area(label='flower description', value=doc)
  
  timgs=[]
  imgs = cImgs.query(query_uris='./' + sim.name, include=['data'], n_results=6)
  for img in imgs['data'][0][0]:
    timgs.append(img)


  dimgs = image_select(
      label='select image',
      images= timgs,
      #captions=['caption one','caption two', 'caption three','caption fore', 'caption five'],
    )

  st.image(dimgs, use_column_width='always')
    
    
    
  
  
