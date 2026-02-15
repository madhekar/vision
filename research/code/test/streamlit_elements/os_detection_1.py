
import platform
import chromadb
from pathlib import Path, WindowsPath
import os

def fix_image_paths():
   client = chromadb.PersistentClient(path="./chroma_db")
   collection = client.get_collection(name="image_collection")

   results = collection.get(include=["metadatas"])
   ids = results['ids']
   metadatas = results['metadatas']

   new_metadatas = []
   for meta in metadatas:
        # Example: Replace old root folder with new one
        old_path = meta['image_path']
        new_path = old_path.replace("/old/path/", "/new/path/")
        meta['image_path'] = new_path
        new_metadatas.append(meta)

   # Update ChromaDB with corrected metadata
   collection.update(ids=ids, metadatas=new_metadatas)



def os_specific_path(img_path):
    linux_prefix = "/mnt/zmdata/"
    mac_prefix = "/Users/Share/zmdata/"
    win_prefix = "c:/Users/Public/zmdata/"
    token = "home-media-app"
    n_pth = ""

    platform_system = platform.system()

    # Example of conditional logic based on OS
    if platform_system == "Windows":
        w_img_path = WindowsPath(img_path)
        parts = w_img_path.split(token,1)
        if len(parts) > 1:
            n_pth = win_prefix + token + parts[1]
        

    elif platform_system == "Linux":
        parts = img_path.split(token,1)
        if len(parts) > 1:
            n_pth = linux_prefix + token + parts[1]
        

    elif platform_system == "Darwin":
        parts = img_path.split(token,1)
        print(parts)
        if len(parts) > 1:
            n_pth = mac_prefix + token + parts[1]

    return n_pth


if __name__=="__main__":
    img_path = "/home/madhekar/work/home-media-app/data/final-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/IMG_6683.PNG"
    print(os_specific_path(img_path=img_path))


