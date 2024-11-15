import os
import json
import uuid
import streamlit as st
from streamlit_tree_select import tree_select
import streamlit_scrollable_textbox as stx

st.set_page_config(
        page_title="zesha: Home Media Portal (HMP)",
        page_icon="/home/madhekar/work/zsource/zesha-high-resolution-logo.jpeg",
        initial_sidebar_state="auto",
        layout="wide",
    )

st.title("🐙 Streamlit-tree-select")
st.subheader("A simple and elegant checkbox tree for Streamlit.")

@st.cache_resource
def path_dict(path):
    d = {"label": os.path.basename(path), "value": str(path) + '@@' + str(uuid.uuid4())}
    if os.path.isdir(path):
        d["label"] = os.path.basename(path)  #'dir'
        d["children"] = [ path_dict(os.path.join(path, x)) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    else:
         d["label"] = os.path.basename(path)  #"file"
    return d      

@st.cache_resource
def path_dict_file(path):
    d1 = {"label": os.path.basename(path), "value": str(path) + '@@' + str(uuid.uuid4())}
    print(path)
    if os.path.isdir(path):
        d1["label"] = os.path.basename(path)  #'dir'
        d1["children"] = [ path_dict_file(os.path.join(path, x)) for x in os.listdir(path)]
    else:
         d1["label"] = os.path.basename(path)  #"file"
    return d1

nodes = []
nodes.append(path_dict("/home/madhekar/work/home-media-app/data/raw-data"))

con = st.container(border=True, height=1000)
#print(nodes)
c1,c2 = con.columns([.5,1], gap='small', vertical_alignment='top')

with c1:
    if nodes:
        return_select = tree_select(nodes, no_cascade=True)
        # stx.scrollableTextbox(return_select, border=True,height = 300)
        #print(return_select['checked'])

with c2:    
    print(return_select['checked'])   
    selected = []
    for e in return_select['checked']:
        e0 = e.split("@@")[0]
        print('->',e)
        selected.append(path_dict_file(e0))
        tree_select(selected)
    #st.write(selected)    
