import os

'''
this module could be expanded as we discover other unknown files
'''
def trim_unknown_files(image_path):
    cnt=0
    mac_file_pattern = "."
    for root, dirs, files in os.walk(image_path):
        for file in files:
            if file.startswith(mac_file_pattern):
                try:
                    os.remove(os.path.join(root,file))
                    print(f"{root} : {dirs} : {file}")
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
                print(path)
                cnt +=1
                os.rmdir(path)
            except OSError as e:
                print(f'exception: {e} removing empty folder {path}')    
    return cnt        

print(trim_unknown_files("/home/madhekar/work/home-media-app/data/input-data-1"))                

print(remove_empty_folders("/home/madhekar/work/home-media-app/data/input-data-1"))
