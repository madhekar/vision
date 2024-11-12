import os
import json
import uuid
import streamlit as st
from streamlit_tree_select import tree_select
import streamlit_scrollable_textbox as stx

st.title("🐙 Streamlit-tree-select")
st.subheader("A simple and elegant checkbox tree for Streamlit.")

mset = set()
d = {}
def path_dict(path):
    # v = os.path.basename(path)
    # if v not in mset:
    #   d["value"] = v
    #   mset.add(v)
    d = {"value": os.path.basename(path)}
    print(path)
    if os.path.isdir(path):
        d["label"] = os.path.basename(path)  #'dir'
        d["children"] = [
            path_dict(os.path.join(path, x)) for x in os.listdir(path)
        ]
    else:
        d["label"] = os.path.basename(path)  #"file"
    print(d)
    return d      

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

print(nodes)
if nodes:
  return_select = tree_select(nodes)
  #stx.scrollableTextbox(return_select, border=True,height = 300)
  st.write(return_select)


