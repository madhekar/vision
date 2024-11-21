import os
import collections
import shutil
import pandas as pd

def count_file_types(folder_path):
    file_counts = collections.defaultdict(int)
    file_sizes = collections.defaultdict(int)
    for  root, _, files in os.walk(folder_path, topdown=True):
        for file in files:
            _, ext = os.path.splitext(file) 
            file_counts[ext] += 1
            file_sizes[ext] += os.stat(os.path.join(root, file)).st_size      
    return file_counts, file_sizes

