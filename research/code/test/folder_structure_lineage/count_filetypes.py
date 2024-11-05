import os
import collections
import pandas as pd

image_types = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.gif', '.GIF', '.bmp', '.BMP', '.tiff', '.TIFF', '.heic','.HEIC']
video_types = []
audio_types = []
document_types = []

def get_size(size):
    if size < 1024:
        return f"{size} bytes"
    elif size < pow(1024,2):
        return f"{round(size/1024, 2)} KB"
    elif size < pow(1024,3):
        return f"{round(size/(pow(1024,2)), 2)} MB"
    elif size < pow(1024,4):
        return f"{round(size/(pow(1024,3)), 2)} GB"

def count_file_types(directory):
    file_counts = collections.defaultdict(int)
    file_sizes = collections.defaultdict(int)
    for  root, _, files in os.walk(directory, topdown=True):
        print (files)
        for file in files:
            _, ext = os.path.splitext(file) #os.stat(file).st_size
            file_counts[ext] += 1
            file_sizes[ext] += os.stat(os.path.join(root, file)).st_size      
    return file_counts, file_sizes

def get_all_image_types(file_counts, file_sizes):
    images_cnt = {key: file_counts[key] for key in image_types if file_counts[key] > 0}
    images_size = {key: get_size(file_sizes[key]) for key in image_types if file_sizes[key] > 0}
    return images_cnt, images_size   

def get_all_video_types(file_counts):
    videos = {key: file_counts[key] for key in video_types if file_counts[key] > 0}
    return videos   

def get_all_file_types(directory_path):        
    file_sizes, file_counts = count_file_types(directory_path)    
    image_cnt, image_sz = get_all_image_types(file_sizes, file_counts)
    
    return pd.DataFrame(image_cnt.items()), pd.DataFrame(image_sz.items())

if __name__ == '__main__':
   df,dfsz = get_all_file_types('/Users/bhal/Downloads')
   print(df.head())
   print(dfsz.head())