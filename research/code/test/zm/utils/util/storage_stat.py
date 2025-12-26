import glob

import os
import collections
import shutil
from PIL import Image
import pandas as pd
import time
import streamlit as st
from tqdm import tqdm
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
    
    if mdf.index.size > 0:
        tot = mdf.shape[0]
        clat = mdf[mdf['GPSLatitude'] == "-"].shape[0]
        clon = mdf[mdf['GPSLatitude'] == "-"].shape[0]
        lat_lon = clat if clat > clon else clon
        cdatetime = mdf[mdf['DateTimeOriginal'] == "-"].shape[0]
        correct = mdf[(mdf['DateTimeOriginal'] != "-") & (mdf['GPSLatitude'] != "-")].shape[0]
    else:
        tot, lat_lon, cdatetime, correct = 0, 0, 0, 0
        
    return {
        "missing-datetime": cdatetime,
        "missing-lat-lon": lat_lon,
        "no-missing-data": correct,
        "total-img-data": tot
    }


def get_folder_metrics(folder_path):
    """
    Calculates the total file count and size (in bytes) for a given folder path.

    Args:
        folder_path (str or Path): The path to the folder.

    Returns:
        tuple: (file_count, total_size_bytes)
    """
    file_count = 0
    total_size_bytes = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Skip if it is a symbolic link to avoid issues
            if not os.path.islink(fp):
                try:
                    total_size_bytes += os.path.getsize(fp)
                    file_count += 1
                except OSError:
                    # Handle cases where file might be inaccessible
                    print(f"Error accessing file: {fp}")
                    pass
    return file_count, total_size_bytes

def bytes_to_gb(bytes_size):
    """Converts bytes to gigabytes."""
    return bytes_size / (1024**3)

def acquire_overview_data(folder_list):

    # Define the list of folders to process
    # folders_to_check = [
    #     "/home/madhekar/work/home-media-app/data/input-data/img",
    #     "/home/madhekar/work/home-media-app/data/input-data/video",
    #     "/home/madhekar/work/home-media-app/data/input-data/txt",
    #     "/home/madhekar/work/home-media-app/data/input-data/audio",
    #     "/home/madhekar/work/home-media-app/data/input-data/error/img/duplicate",
    #     "/home/madhekar/work/home-media-app/data/input-data/error/img/missing-data",
    #     "/home/madhekar/work/home-media-app/data/input-data/error/img/quality",
    #     "/home/madhekar/work/home-media-app/data/input-data/error/video/duplicate",
    #     "/home/madhekar/work/home-media-app/data/input-data/error/video/missing-data",
    #     "/home/madhekar/work/home-media-app/data/input-data/error/video/quality",
    #     "/home/madhekar/work/home-media-app/data/input-data/error/txt/duplicate",
    #     "/home/madhekar/work/home-media-app/data/input-data/error/txt/missing-data",
    #     "/home/madhekar/work/home-media-app/data/input-data/error/txt/quality",
    #     "/home/madhekar/work/home-media-app/data/input-data/error/audio/duplicate",
    #     "/home/madhekar/work/home-media-app/data/input-data/error/audio/missing-data",
    #     "/home/madhekar/work/home-media-app/data/input-data/error/audio/quality",
    #     "/home/madhekar/work/home-media-app/data/final-data/img",
    #     "/home/madhekar/work/home-media-app/data/final-data/video",
    #     "/home/madhekar/work/home-media-app/data/final-data/txt",
    #     "/home/madhekar/work/home-media-app/data/final-data/audio",
    # ]

    print(f"{'Folder':<30} | {'File Count':<15} | {'Size (GB)':<15}")
    print("-" * 64)

    src_list, r_list = ['madhekar', 'Samsung USB'], []
    prefix = "/home/madhekar/work/home-media-app/data/"
    for src in src_list:
        for folder in folder_list:
            folder = os.path.join(folder, src)
            print(f"--->>{folder}")
            if os.path.isdir(folder):
                count, size_bytes = get_folder_metrics(folder)
                size_gb = bytes_to_gb(size_bytes)
                f_trim = folder.removeprefix(prefix)
                f_trim = f_trim.replace("/error","")
                # print(f_trim)
                n_path = os.path.normpath(f_trim)
                path_list = n_path.split(os.sep)
                if len(path_list) ==3:
                    path_list[2] = "data"
                print(path_list)
                # print(f"{f_trim:<30} | {count:<15} | {size_gb:<15.4f}")
                r_list.append({"source": src, "data_stage": path_list[0], "data_type": path_list[1], "data_attrib": path_list[2],  "count": count, "size": size_gb})

            else:
                
                f_trim = folder.removeprefix(prefix)
                f_trim = f_trim.replace("/error", "")
                print(f_trim)
                n_path = os.path.normpath(f_trim)
                path_list = n_path.split(os.sep)
                if len(path_list) == 3:
                    path_list[2] = "data"
                print(f'--path list-> {path_list}')
                # print(f"{f_trim:<30} | {0:<15} | {0:<15.4f} ")
                r_list.append({"source": src, "data_stage": path_list[0], "data_type": path_list[1], "data_attrib": path_list[2],  "count": 0, "size": 0.0})
                pass

    df = pd.DataFrame(r_list, columns=["source", "data_stage", "data_type", "data_attrib", "count", "size"])
    print(df)

    values_to_delete = ['duplicate','missing-data','quality']
    dft = df[~((df['data_stage'] == "final-data") & (df['data_attrib'].isin(values_to_delete)))]
    print(dft)

    df_input = dft[~(dft['data_stage'] == "final-data")]
    df_final = dft[~(dft['data_stage'] == "input-data")]
    # out = dft.pivot_table(index=["source", "data_stage", "data_type"], columns=["data_attrib"], values=["count", "size"])
    # print(out)
    #dft = dft[~(dft['source'] == "Samsung USB")]
    return df_input, df_final


"""
this module could be expanded as we discover 
other unknown files
"""
def trim_unknown_files(image_path):
    cnt = 0
    mac_file_pattern = "._"
    # total files
    total_items = 0
    for dp, dns, fns in os.walk(image_path):
        total_items += len(dns) + len(fns)

    with tqdm(total=total_items, desc=f'trim unknown files: {image_path}') as pbar:
        for root, dirs, files in os.walk(image_path):
            for file in files:
                if file.startswith(mac_file_pattern):
                    try:
                        os.remove(os.path.join(root, file))
                        #print(f'removed {os.path.join(root, file)}')
                        cnt += 1
                    except OSError as e:
                        print(f"exception: {e} removing empty file {file}.")
                pbar.update(1)        
    return cnt

def remove_all_files_by_type(root_folder, type):
    cnt = 0
    dtype = image_types if type == 'I' else video_types
    for file in os.listdir(root_folder):
            if os.path.splitext(file)[1].lower() in dtype:
                try:
                    os.remove(os.path.join(root_folder, file))
                    #print(f'removed {os.path.join(root_folder, file)}')
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

def remove_empty_image_files_and_folders(root_folder):
    fc, dc = 0, 0
    mac_file_pattern = '._'
    for dpath, dnames, files in os.walk(root_folder, topdown=False):
        # empty files
        for fn in files:
            fp = os.path.join(dpath, fn)
            try:  
                f = Image.open(fp)
                f.verify()
                f.close()
                if os.path.getsize(fp) < 512 or fn.startswith(mac_file_pattern):
                    try:
                        os.remove(fp)
                        fc +=1
                        #print(f'removed empty file: {fp}')
                    except (OSError, SyntaxError) as e:
                        print(f'error removing file {fp}: {e}')  
            except (OSError, SyntaxError) as e:
                print(f'bad or corrupt file{fp}: {e}')
                os.remove(fp)
             
        if not os.listdir(dpath):
            try:
                os.rmdir(dpath)
                dc +=1
                #print(f'removed empty folder: {dpath}')
            except OSError as e:
                print(f'error removing folder{dpath}: {e}')    
    return dc, fc

def remove_file(file_path):
    if os.path.exists(file_path):
        if os.path.getsize(file_path) < 50:
            os.remove(file_path)
        else:
            print(f'file not empty: {file_path}')    
    else:
        print(f'file not found {file_path}')        

# get immediate child folders
def extract_user_raw_data_folders(pth):
    return next(os.walk(pth))[1]

def create_folder(cpath):
    try:
        os.makedirs(cpath, exist_ok=True)
    except OSError as e:
        print(f' Error  creating folder {cpath} : {e}')    

def worker_function(item):
    """A function to be executed in a separate process."""
    time.sleep(0.1)  # Simulate some work
    return item * 2        

if __name__ == '__main__':
    extract_all_folder_stats("/home/madhekar/work/home-media-app/data/raw-data")
    extract_server_stats()
    #extract_folderlist_stats(["/home/madhekar/work/home-media-app/data/raw-data/AnjaliBackup", '/home/madhekar/work/home-media-app/data/raw-data/Madhekar'])
