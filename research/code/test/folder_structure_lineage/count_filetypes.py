import os
import collections
import shutil
import pandas as pd

image_types = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.gif', '.GIF', '.bmp', '.BMP', '.tiff', '.TIFF', '.heic','.HEIC']
video_types = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']
audio_types = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
document_types = [".txt", ".doc", ".docx", ".pdf", ".xls", ".xlsx", ".ppt", ".pptx"]

def get_size(size):
    # if size < 1024:
    #     return f"{size} bytes"
    # elif size < pow(1024,2):
    #     return f"{round(size/1024, 2)} KB"
    # elif size < pow(1024,3):
    #     return f"{round(size/(pow(1024,2)), 2)} MB"
    # elif size < pow(1024,4):
    #     return f"{round(size/(pow(1024,3)), 2)} GB"
    return round(size/(pow(1024,2)), 2)
    
def get_dataframe(cnt, size):
    df = pd.DataFrame.from_dict([cnt, size])
    new_columns = {0: 'count', 1: 'size'}
    df = df.T
    df.rename(columns=new_columns, inplace=True)
    return df

def count_file_types(directory):
    file_counts = collections.defaultdict(int)
    file_sizes = collections.defaultdict(int)
    for  root, _, files in os.walk(directory, topdown=True):
        for file in files:
            _, ext = os.path.splitext(file) #os.stat(file).st_size
            file_counts[ext] += 1
            file_sizes[ext] += os.stat(os.path.join(root, file)).st_size      
    return file_counts, file_sizes

def get_all_image_types(file_counts, file_sizes):
    images_cnt = {key: file_counts[key] for key in image_types if file_counts[key] > 0}
    images_size = {key: get_size(file_sizes[key]) for key in image_types if file_sizes[key] > 0}
    return images_cnt, images_size   

def get_all_video_types(file_counts, file_sizes):
    videos_cnt = {key: file_counts[key] for key in video_types if file_counts[key] > 0}
    videos_size = {key: get_size(file_sizes[key]) for key in video_types if file_sizes[key] > 0}
    return  videos_cnt, videos_size  

def get_all_document_types(file_counts, file_sizes):
    documents_cnt = {key: file_counts[key] for key in document_types if file_counts[key] > 0}
    documents_size = {key: get_size(file_sizes[key]) for key in document_types if file_sizes[key] > 0}
    return  documents_cnt, documents_size  

def get_all_audio_types(file_counts, file_sizes):
    audios_cnt = {key: file_counts[key] for key in audio_types if file_counts[key] > 0}
    audios_size = {key: get_size(file_sizes[key]) for key in audio_types if file_sizes[key] > 0}
    return  audios_cnt, audios_size  

def get_all_file_types(directory_path):        
    file_counts, file_sizes = count_file_types(directory_path)
    dfi = get_dataframe(*get_all_image_types(file_counts, file_sizes))
    dfv = get_dataframe(*get_all_video_types(file_counts, file_sizes))
    dfd = get_dataframe(*get_all_document_types(file_counts, file_sizes))
    dfa = get_dataframe(*get_all_audio_types(file_counts, file_sizes))
    return dfi, dfv, dfd, dfa

def get_disk_usage():
    total, used, free = shutil.disk_usage("/")
    return (total, used, free)
if __name__ == '__main__':
   dfi, dfv,dfd,dfa = get_all_file_types('/Users/bhal/Downloads')
   if not dfi.empty:
     print(dfi.head())
     print('Total files:', dfi['count'].sum(), 'Total Size (MB): ', dfi['size'].sum())

   if not dfv.empty:  
     print(dfv.head())
     print('Total files:', dfv['count'].sum(), 'Total Size (MB): ', dfv['size'].sum())

   if not dfd.empty:  
     print(dfd.head())
     print('Total files:', dfd['count'].sum(), 'Total Size (MB): ', dfd['size'].sum())

   if not dfa.empty:  
     print(dfa.head())
     print('Total files:', dfa['count'].sum(), 'Total Size (MB): ', dfa['size'].sum())

total, used, free = get_disk_usage()
print("Total Size:", total // (2**30),"GB", "Used Size:",used // (2**30), "GB", "Free Size: ", free // (2**30), "GB")     
