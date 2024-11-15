import os
import json
import uuid
import streamlit as st
from streamlit_tree_select import tree_select
import streamlit_scrollable_textbox as stx

st.title("🐙 Streamlit-tree-select")
st.subheader("A simple and elegant checkbox tree for Streamlit.")


@st.cache_resource
def path_dict(path):
    d = {"label": os.path.basename(path), "value": str(path) + '@@' + str(uuid.uuid4())} #, 'id': str(uuid.uuid4())}
    if os.path.isdir(path):
        d["label"] = os.path.basename(path)  #'dir'
        d["children"] = [
            path_dict(os.path.join(path, x))
            for x in os.listdir(path)
            if os.path.isdir(os.path.join(path, x))
        ]
    # else:
    #     d["label"] = os.path.basename(path)  #"file"
    #print(d)
    return d      

@st.cache_resource
def path_dict_file(path):
    d1 = {"label": os.path.basename(path), "value": str(path) + '@@' + str(uuid.uuid4())} #, 'id': str(uuid.uuid4())}
    print(path)
    if os.path.isdir(path):
        d1["label"] = os.path.basename(path)  #'dir'
        d1["children"] = [
            path_dict_file(os.path.join(path, x))
            for x in os.listdir(path)
        ]
    else:
         d1["label"] = os.path.basename(path)  #"file"
    print(d1)
    return d1


def dict_to_json(path):
   return json.dumps(path_dict(path),0)

# Create nodes to display
# nodes = [
#     {"label": "Folder A", "value": "folder_a"},
#     {
#         "label": "Folder B",
#         "value": "folder_b",
#         "children": [
#             {"label": "Sub-folder A", "value": "sub_a"},
#             {"label": "Sub-folder B", "value": "sub_b"},
#             {"label": "Sub-folder C", "value": "sub_c"},
#         ],
#     },
#     {
#         "label": "Folder C",
#         "value": "folder_c",
#         "children": [
#             {"label": "Sub-folder D", "value": "sub_d"},
#             {
#                 "label": "Sub-folder E",
#                 "value": "sub_e",
#                 "children": [
#                     {"label": "Sub-sub-folder A", "value": "sub_sub_a"},
#                     {"label": "Sub-sub-folder B", "value": "sub_sub_b"},
#                 ],
#             },
#             {"label": "Sub-folder F", "value": "sub_f"},
#         ],
#     },
# ]
#dict_to_json('/home/madhekar/work/home-media-path/data/raw-data')
nodes = []
nodes.append(path_dict("/home/madhekar/work/home-media-app/data/raw-data"))

#print(nodes)
c1,c2 = st.columns([.5,1], gap='small', vertical_alignment='top')

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
