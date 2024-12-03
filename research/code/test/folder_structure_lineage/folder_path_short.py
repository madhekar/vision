import uuid
import os
import hashlib
import shutil


media_extensions = ['.mp3', '.mp4', '.avi', '.mov', '.jpg', '.jpeg', '.png', '.gif']  # Add more as needed
document_extensions = ['.txt', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.log']  # Add more as needed

def create_uuid_from_string(val: str):
   hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
   return uuid.UUID(hex=hex_string)

def path_encode(spath):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, spath))

def copy_files_only(src_dir, dest_dir):

    if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir, ignore_errors=True)
            # making the destination directory
            os.makedirs(dest_dir)

    for root, dirnames, items in os.walk(src_dir):
        if not dirnames:
            if len(items) > 0:
                print(root + " - " + str(dirnames) + " - " + str(items))
                items = [ f for f in items if os.path.splitext(f)[1].lower() in media_extensions or os.path.splitext(f)[1].lower() in document_extensions]
                if len(items) > 0:
                    print(root + " - " + str(dirnames) + " - " + str(items))
                    uuid_path = path_encode(root)
                    f_dest = os.path.join(dest_dir, uuid_path)
                    print(src_dir + " - " + f_dest +" - " + str(len(items)))
                    os.makedirs(f_dest)
                    for item in items:
                        item_path = os.path.join(root, item)
                        print(item_path+ ' -> '+ dest_dir)
                        if os.path.isfile(item_path):
                            print('**' + item_path + " - "+ dest_dir)
                            try:
                                shutil.copy(item_path, f_dest)
                            except FileNotFoundError:
                                print("Source file not found.")
                            except PermissionError:
                                print("Permission denied.")
                            except FileExistsError:
                                print("Destination file already exists.")
                            except Exception as e:
                                print(f"An error occurred: {e}")
                    

# def create_dirtree_without_files(src, dst):
   
#     # getting the absolute path of the source
#     # directory
#     src = os.path.abspath(src)
#     #print(dst) 
#     # making a variable having the index till which
#     # src string has directory and a path separator
#     src_prefix = len(src) + len(os.path.sep)
#     #dst_path = os.path.join(dst, '/')
     
#     if os.path.exists(dst):
#         shutil.rmtree(dst, ignore_errors=True) 
#         # making the destination directory
#         os.makedirs(dst)


     
#     # doing os walk in source directory
#     for root, dirs, files in os.walk(src):
#         #for dirname in dirs:
#         if not dirs:   
#             # here dst has destination directory, 
#             # root[src_prefix:] gives us relative
#             # path from source directory 
#             # and dirname has folder names
#             dirpath = os.path.join(dst, root[src_prefix:], dirname)
#             srcpath = os.path.join(src, dirname)
#             #print(srcpath) 
#             # making the path which we made by
#             # joining all of the above three
#             if len(files) > 0:
#                 uuid_path = path_encode(dirpath)
#                 f_dest = os.path.join(dst, uuid_path)
#                 print(srcpath + " - " + f_dest +" - " + str(len(files)))
#                 os.mkdir(f_dest)
#                 copy_files_only(srcpath, f_dest)
 

# def path_encode_1(spath):
#   return str(uuid.uuid5(str.encode('zesha llc'),spath))

plist = [
    "/home/madhekar/work/home-media-portal/data/raw-data/1",
    "/home/madhekar/work/home-media-portal/data/raw-data/5",
    "/home/madhekar/work/home-media-portal/data/raw-data/0",
    "/home/madhekar/work/home-media-portal/data/raw-data/",
    "/home/madhekar/work/home-media-portal/data/raw-data",
    "/home",
    "",
    "/"
]

# print({x: path_encode(x) for x in plist})

# print({x: path_encode_1(x) for x in plist})


# print({x: str(create_uuid_from_string(x)) for x in plist})


copy_files_only('/home/madhekar/work/home-media-app/data/raw-data/Madhekar/',
                             '/home/madhekar/temp/img_backup/ex1')
