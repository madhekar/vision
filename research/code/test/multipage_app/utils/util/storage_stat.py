import glob

import os
import collections
import shutil
import pandas as pd

image_types = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.gif', '.GIF', '.bmp', '.BMP', '.tiff', '.TIFF', '.heic','.HEIC']
video_types = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']
audio_types = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
document_types = [".txt", ".doc", ".docx", ".pdf", ".xls", ".xlsx", ".ppt", ".pptx"]

class FolderStats:
    def __init__(self, fpath):
       self.folder_path = fpath

    def get_size(self, size):
        if size < 1024:
            return f"{size} bytes"
        elif size < pow(1024,2):
            return f"{round(size/1024, 2)} KB"
        elif size < pow(1024,3):
            return f"{round(size/(pow(1024,2)), 2)} MB"
        elif size < pow(1024,4):
            return f"{round(size/(pow(1024,3)), 2)} GB"
    
    def get_size_for_plot(self, size):
        return round(size / (pow(1024, 2)), 2)
        
    def get_dataframe(self, cnt, size):
        df = pd.DataFrame.from_dict([cnt, size])
        new_columns = {0: 'count', 1: 'size'}
        df = df.T
        df.rename(columns=new_columns, inplace=True)
        return df

    def count_file_types(self):
        file_counts = collections.defaultdict(int)
        file_sizes = collections.defaultdict(int)
        for  root, _, files in os.walk(self.folder_path, topdown=True):
            for file in files:
                _, ext = os.path.splitext(file) 
                file_counts[ext] += 1
                file_sizes[ext] += os.stat(os.path.join(root, file)).st_size      
        return file_counts, file_sizes

    def get_all_image_types(self, file_counts, file_sizes):
        images_cnt = {key: file_counts[key] for key in image_types if file_counts[key] > 0}
        images_size = {key: self.get_size_for_plot(file_sizes[key]) for key in image_types if file_sizes[key] > 0}
        return images_cnt, images_size   

    def get_all_video_types(self, file_counts, file_sizes):
        videos_cnt = {key: file_counts[key] for key in video_types if file_counts[key] > 0}
        videos_size = {key: self.get_size_for_plot(file_sizes[key]) for key in video_types if file_sizes[key] > 0}
        return  videos_cnt, videos_size  

    def get_all_document_types(self, file_counts, file_sizes):
        documents_cnt = {key: file_counts[key] for key in document_types if file_counts[key] > 0}
        documents_size = {key: self.get_size_for_plot(file_sizes[key]) for key in document_types if file_sizes[key] > 0}
        return  documents_cnt, documents_size  

    def get_all_audio_types(self, file_counts, file_sizes):
        audios_cnt = {key: file_counts[key] for key in audio_types if file_counts[key] > 0}
        audios_size = {key: self.get_size_for_plot(file_sizes[key]) for key in audio_types if file_sizes[key] > 0}
        return  audios_cnt, audios_size  

    def get_all_file_types(self):        
        file_counts, file_sizes = self.count_file_types()
        dfi = self.get_dataframe(*self.get_all_image_types(file_counts, file_sizes))
        dfv = self.get_dataframe(*self.get_all_video_types(file_counts, file_sizes))
        dfd = self.get_dataframe(*self.get_all_document_types(file_counts, file_sizes))
        dfa = self.get_dataframe(*self.get_all_audio_types(file_counts, file_sizes))
        return dfi, dfv, dfd, dfa

    def get_disk_usage(self):
        total, used, free = shutil.disk_usage("/")
        return (total, used, free)
    
def execute(folder_path):
   fstat = FolderStats(folder_path)  

   dfi, dfv,dfd,dfa = fstat.get_all_file_types()
   if not dfi.empty:
     print(dfi.head())
     print('Image - Total files:', dfi['count'].sum(), 'Total Size (MB): ', dfi['size'].sum())

   if not dfv.empty:  
     print(dfv.head())
     print('Video - Total files:', dfv['count'].sum(), 'Total Size (MB): ', dfv['size'].sum())

   if not dfd.empty:  
     print(dfd.head())
     print('Document - Total files:', dfd['count'].sum(), 'Total Size (MB): ', dfd['size'].sum())

   if not dfa.empty:  
     print(dfa.head())
     print('Audio - Total files:', dfa['count'].sum(), 'Total Size (MB): ', dfa['size'].sum())

   total, used, free = fstat.get_disk_usage()
   print("Total Size:", total // (2**30),"GB", "Used Size:",used // (2**30), "GB", "Free Size: ", free // (2**30), "GB")   

if __name__ == '__main__':
   
   execute('/home/madhekar/work/home-media-app/data/raw-data')
