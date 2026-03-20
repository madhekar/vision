import chromadb
import pandas as pd
import streamlit as st

from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report


def get_vdb_connection(vdb_path):
    client = None
    try:
      client = chromadb.PersistentClient(vdb_path)
      print(f"number of collections found: {client.count_collections()}")  
    except Exception as e:
        print(f"exception occured getting vdb client connection: {e}")  
    return client    

def get_collection_record_count(vdb_list):

    vdb_path =  vdb_list.pop(0)

    print(vdb_list)

    client = get_vdb_connection(vdb_path=vdb_path)

    collection_count = []
   
    for c in vdb_list: 
        try:
            cc = client.get_collection(name=c)
            val = cc.count()
            collection_count.append({"modality": c.removeprefix("multimodal_collection_"), 'count': val})
        except Exception as e:
            print(f"excption occured while getting collection: {c} as: {e}")    
            collection_count.append({"modality": c.removeprefix("multimodal_collection_"), 'count': 0})
            continue
    print(collection_count)    
    return  collection_count    


vdb_list = ["/mnt/zmdata/home-media-app/data/app-data/vectordb/", "multimodal_collection_images", "multimodal_collection_texts", "multimodal_collection_videos", "multimodal_collection_audios"]
get_collection_record_count(vdb_list=vdb_list)

# num_collections = client.count_collections()

# print(f"collections: {num_collections}")

# icollection = client.get_collection(name="multimodal_collection_images")
# tcollection = client.get_collection(name="multimodal_collection_texts")
# try:
#     vcollection = client.get_collection(name="multimodal_collection_videos")
#     acollection = client.get_collection(name="multimodal_collection_audios")
#     print(f"count records: {vcollection.count()} name: {vcollection.name} ")
#     print(f"count records: {acollection.count()} name: {acollection.name} ")
# except   Exception as e:
#     print(e)

# print(f"count records: {tcollection.count()} name: {tcollection.name} ")
# print(f"count records: {icollection.count()} name: {icollection.name} ")
# # Retrieve all data (be mindful of large collections)
# iall_data = icollection.get(
#     include=["metadatas", "embeddings", "documents"]
# )

# tall_data = tcollection.get(include=["metadatas", "documents", "embeddings"])
# print('-->', tall_data["ids"], "::", tall_data["embeddings"])
# #print(all_data["ids"], all_data["uris"])

# df = pd.DataFrame(iall_data['metadatas'])
# df['ids'] = iall_data["ids"]
# print(df.describe(include='all').transpose())
# pr = df.profile_report(title="My Custom Profiling Report",minimal=True,explorative=False)
# st_profile_report(pr)
# # Convert metadata to a pandas DataFrame        
# # This assumes your metadata has numerical fields you want to analyze
# if all_data['metadatas']:
#     df = pd.DataFrame(all_data['metadatas'])
    
#     # Perform descriptive statistics on the DataFrame
#     print("\nDescriptive statistics for metadata:")
#     print(df["loc"].value_counts().head(10))
#     print(df['loc'].unique())
#     #print(df['loc'].value_counts())
#     print(df.describe()) # .describe() works well on numeric columns

# print(df.describe().transpose())
# # You can also analyze the length of documents or other data points as needed
# document_lengths = [len(doc) for doc in all_data['documents']]
# df_lengths = pd.DataFrame(document_lengths, columns=['document_length'])
# print("\nDescriptive statistics for document lengths:")
# print(df_lengths.describe())
