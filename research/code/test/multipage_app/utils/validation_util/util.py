import os
import shutil
import pandas as pd

""" 

"""
def replicate_folders_files(root_src_dir, root_dst_dir):
    directories_added = []
    files_added = []
    memory_used = []

    memory_used.append(shutil.disk_usage(root_dst_dir))
    for src_dir, dirs, files in os.walk(root_src_dir):
        print(src_dir, dirs, files)
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            directories_added.append(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                # in case of the src and dst are the same file
                # if os.path.samefile(src_file, dst_file):
                continue
            # os.remove(dst_file)
            shutil.copy(src_file, dst_dir)
            files_added.append(dst_file)

    memory_used.append(shutil.disk_usage(root_dst_dir))

    return (directories_added, files_added, memory_used)

'''
idx, state, total_memory, used, free
'''
def update_audit_records(audit_path, audit_file_name):

    if (os.path.exists(os.path.join(audit_path, audit_file_name))):
        df = pd.read.csv(os.path.join(audit_path, audit_file_name))
        return df
