import os
import streamlit as st
import uuid

@st.cache_resource
def path_to_dict(path):
    d = {
        "label": os.path.basename(path),
        "value": str(path) + "@@" + str(uuid.uuid4()),
    }  # , 'id': str(uuid.uuid4())}
    if os.path.isdir(path):
        d["label"] = os.path.basename(path)  #'dir'
        d["children"] = [
            path_to_dict(os.path.join(path, x))
            for x in os.listdir(path)
            if os.path.isdir(os.path.join(path, x))
        ]
    # else:
    #     d["label"] = os.path.basename(path)  #"file"
    # print(d)
    return d


def path_dict_file(path):
    d1 = {
        "label": os.path.basename(path),
        "value": str(path) + "@@" + str(uuid.uuid4()),
    }  # , 'id': str(uuid.uuid4())}
    print(path)
    if os.path.isdir(path):
        d1["label"] = os.path.basename(path)  #'dir'
        d1["children"] = [path_dict_file(os.path.join(path, x)) for x in os.listdir(path)]
    else:
        d1["label"] = os.path.basename(path)  # "file"
    print(d1)
    return d1
