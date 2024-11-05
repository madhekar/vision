import os
import collections
import pandas as pd

image_types = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.gif', '.GIF', '.bmp', '.BMP', '.tiff', '.TIFF']
video_types = []
audio_types = []
document_types = []

def count_file_types(directory):
    file_counts = collections.defaultdict(int)
    file_sizes = collections.defaultdict(int)
    for root, _, files in os.walk(directory, topdown=True):
        for file in files:
            _, ext = os.path.splitext(file) #os.stat(file).st_size
            file_counts[ext] += 1
            file_sizes[ext] += 45
    return file_counts, file_sizes

def get_all_image_types(file_counts, file_sizes):
    images_cnt = {key: file_counts[key] for key in image_types if file_counts[key] > 0}
    images_size = {key: file_sizes[key] for key in image_types if file_sizes[key] > 0}
    return images_cnt, images_size   

def get_all_video_types(file_counts):
    videos = {key: file_counts[key] for key in video_types if file_counts[key] > 0}
    return videos   

def get_all_file_types(directory_path):        
    file_sizes, file_counts = count_file_types(directory_path)    
    image_cnt, image_sz = get_all_image_types(file_sizes, file_counts)
    return pd.DataFrame(image_cnt.items()), pd.DataFrame(image_sz.items())


if __name__ == '__main__':
   df,dfsz = get_all_file_types('/home/madhekar/work/home-media-app/data/input-data/')
   print(df.head())
   print(dfsz.head())