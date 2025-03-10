import os

def trim_empty_folders(image_path):
    removed_cnt=0
    for root, dirs, files in os.walk(image_path):
        for file in files:
           if file.startswith('._'):
                try:
                    #os.rmdir(dir_path)
                    print(f"{root} : {dirs} : {file}")
                    removed_cnt += 1
                except OSError as e:
                    print( f"exception: {e} removing empty folder {root}.")
    return removed_cnt
print(trim_empty_folders("/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup"))                

