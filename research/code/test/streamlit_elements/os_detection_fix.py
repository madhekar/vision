import platform
import chromadb
from pathlib import Path, WindowsPath
import os
cdb_pth = "/mnt/zmdata/home-media-app/data/app-data/vectordb/"
img_idx = "multimodal_collection_images"

def os_specific_prefix():
    linux_prefix = "/mnt/zmdata/"
    mac_prefix = "/Users/Share/zmdata/"
    win_prefix = "c:/Users/Public/zmdata/"

    platform_system = platform.system()

    if platform_system == "Windows":
        return win_prefix
    elif platform_system == "Linux":
        return linux_prefix
    elif platform_system == "Darwin":
        return mac_prefix
    else:
        return ""
    
def fix_uri(img, prefix, token):
    parts = img.split(token,1)
    if len(parts) > 1:
        n_pth = prefix + token + parts[1]
    return n_pth    
   

def fix_image_paths():
   
   token = "home-media-app" 
   prefix = os_specific_prefix()
   ncdb_pth = fix_uri(cdb_pth, prefix=prefix, token=token)
   client = chromadb.PersistentClient(path=ncdb_pth)
   collection = client.get_collection(name=img_idx)
  
   results = collection.get(include=["uris"])
   print(prefix) #, results)
   
   ids = results["ids"]
   uris = results['uris']
   ruris = [fix_uri(uri, prefix=prefix, token=token) for uri in uris]
   print('****', ruris)
   collection.update(ids=ids, uris=ruris)
#    ids = results['ids']
#    uris = results['uris']
#    metadatas = results['metadatas']

#    new_metadatas = []
#    for meta in metadatas:
#         # Example: Replace old root folder with new one
#         old_path = meta['image_path']
#         new_path = old_path.replace("/old/path/", "/new/path/")
#         meta['image_path'] = new_path
#         new_metadatas.append(meta)

#    # Update ChromaDB with corrected metadata
#    collection.update(ids=ids, metadatas=new_metadatas)



# def os_specific_path(img_path):
#     linux_prefix = "/mnt/zmdata/"
#     mac_prefix = "/Users/Share/zmdata/"
#     win_prefix = "c:/Users/Public/zmdata/"
#     token = "home-media-app"
#     n_pth = ""

#     platform_system = platform.system()

#     # Example of conditional logic based on OS
#     if platform_system == "Windows":
#         w_img_path = WindowsPath(img_path)
#         parts = w_img_path.split(token,1)
#         if len(parts) > 1:
#             n_pth = win_prefix + token + parts[1]
        

#     elif platform_system == "Linux":
#         parts = img_path.split(token,1)
#         if len(parts) > 1:
#             n_pth = linux_prefix + token + parts[1]
        

#     elif platform_system == "Darwin":
#         parts = img_path.split(token,1)
#         print(parts)
#         if len(parts) > 1:
#             n_pth = mac_prefix + token + parts[1]

#     return n_pth


if __name__=="__main__":
    # img_path = "/home/madhekar/work/home-media-app/data/final-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/IMG_6683.PNG"
    # print(os_specific_path(img_path=img_path))
    fix_image_paths()
