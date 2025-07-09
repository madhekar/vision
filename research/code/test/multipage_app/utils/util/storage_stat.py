import glob

import os
import collections
import shutil
import pandas as pd
import streamlit as st
from .file_type_ext import image_types, video_types, audio_types, document_types, non_media_types 


@st.cache_resource
class FolderStats:

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
        # print(cnt)
        # df1 = pd.DataFrame.from_dict([cnt]).columns(['type', 'cnt'])
        # df2 = pd.DataFrame.from_dict([size]).columns(["type", "size"])
        # print(df1)
        # df = pd.merge(df1, df2, on='type')
        df = pd.DataFrame.from_dict([cnt, size])
        new_columns = {0: 'count', 1: 'size'}
        df = df.T
        df.rename(columns=new_columns, inplace=True)
        #df['type'] = df.index
        return df

    def count_file_types(self, folder_path):
        file_counts = collections.defaultdict(int)
        file_sizes = collections.defaultdict(int)
        for  root, _, files in os.walk(folder_path, topdown=True):
            for file in files:
                ext = os.path.splitext(file)[1].lower() 
                file_counts[ext] += 1
                file_sizes[ext] += os.stat(os.path.join(root, file)).st_size      
        return file_counts, file_sizes

    def get_all_file_ext_types(self, file_counts, file_sizes):
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
    
    def get_all_non_media_types(self, file_counts, file_sizes):
        non_medias_cnt = {key: file_counts[key] for key in non_media_types if file_counts[key] > 0}
        non_medias_size = {key: self.get_size_for_plot(file_sizes[key]) for key in non_media_types if file_sizes[key] > 0}
        return  non_medias_cnt, non_medias_size  

    def get_all_file_types(self, fpath):        
        file_counts, file_sizes = self.count_file_types(fpath)
        dfx = self.get_dataframe(*self.get_all_file_ext_types(file_counts, file_sizes))
        dfi = self.get_dataframe(*self.get_all_image_types(file_counts, file_sizes))
        dfv = self.get_dataframe(*self.get_all_video_types(file_counts, file_sizes))
        dfd = self.get_dataframe(*self.get_all_document_types(file_counts, file_sizes))
        dfa = self.get_dataframe(*self.get_all_audio_types(file_counts, file_sizes))
        dfn = self.get_dataframe(*self.get_all_non_media_types(file_counts, file_sizes))
        return dfx, dfi, dfv, dfd, dfa, dfn

    def get_all_file_ext_types_by_folder(self, fpath):
        return self.get_dataframe(*self.count_file_types(fpath))
    
    def get_disk_usage(self):
        total, used, free = shutil.disk_usage("/")
        return (total // (2**30), used // (2**30), free // (2**30))
    

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        return True
    if not os.path.isdir(folder_path):
        return False
    if not os.listdir(folder_path):
        return False
    return False    
   
def extract_all_folder_stats(folder_path):
   fstat = FolderStats()  

   dfx, dfi, dfv, dfd, dfa, dfn = fstat.get_all_file_types(folder_path)

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

   if not dfn.empty:  
     print(dfn.head())
     print('Non Media - Total files:', dfn['count'].sum(), 'Total Size (MB): ', dfn['size'].sum())   
     
   if not dfx.empty:
     print(dfx.head())
     print('Image - Total files:', dfx['count'].sum(), 'Total Size (MB): ', dfx['size'].sum())

   return (dfi, dfv, dfd, dfa, dfn)    

def extract_all_file_stats_in_folder(folder_path):
    fstat = FolderStats()

    df = fstat.get_all_file_ext_types_by_folder(folder_path)

    return df

def extract_folder_stats(folder):
    fstat = FolderStats()
    dfe = None
    if not check_folder(folder):
        dfe = fstat.get_all_file_ext_types_by_folder(folder)
        if not dfe.empty:
            return dfe
    return dfe
    
def extract_server_stats():
  fstat = FolderStats()  
  total, used, free = fstat.get_disk_usage()
  print("Total Size:", total ,"GB", "Used Size:",used , "GB", "Free Size: ", free , "GB")  
  return (total , used , free )

def extract_stats_of_metadata_file(metadata_path):
    print(metadata_path)
    mdf = pd.read_csv(metadata_path)
    print(mdf.head(10))
    #tot = mdf.shape[0]
    clat = mdf[mdf['GPSLatitude'] == "-"].shape[0]
    clon = mdf[mdf['GPSLatitude'] == "-"].shape[0]
    lat_lon = clat if clat > clon else clon
    cdatetime = mdf[mdf['DateTimeOriginal'] == "-"].shape[0]
    correct = mdf[(mdf['DateTimeOriginal'] != "-") & (mdf['GPSLatitude'] != "-")].shape[0]

    return {
        #"total": tot,
        "no-missing-data": correct,
        "missging-lat-lon": lat_lon,
       # "lon" : clon,
        "missing-datetime": cdatetime,
       # "latlon_n_datetime": clatlong_n_datetime
    }

"""
this module could be expanded as we discover 
other unknown files
"""
def trim_unknown_files(image_path):
    cnt = 0
    mac_file_pattern = "._"
    for root, dirs, files in os.walk(image_path):
        for file in files:
            if file.startswith(mac_file_pattern):
                try:
                    os.remove(os.path.join(root, file))
                    cnt += 1
                except OSError as e:
                    print(f"exception: {e} removing empty file {file}.")
    return cnt


def remove_empty_folders(path_absolute):
    cnt = 0
    walk = list(os.walk(path_absolute))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            try:
                cnt += 1
                os.rmdir(path)
            except OSError as e:
                print(f"exception: {e} removing empty folder {path}")
    return cnt

# get immediate child folders
def extract_user_raw_data_folders(pth):
    return next(os.walk(pth))[1]


if __name__ == '__main__':
    extract_all_folder_stats("/home/madhekar/work/home-media-app/data/raw-data")
    extract_server_stats()
    #extract_folderlist_stats(["/home/madhekar/work/home-media-app/data/raw-data/AnjaliBackup", '/home/madhekar/work/home-media-app/data/raw-data/Madhekar'])
