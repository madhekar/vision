import uuid
import os
import hashlib

def create_uuid_from_string(val: str):
   hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
   return uuid.UUID(hex=hex_string)

def path_encode(spath):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, spath))

def create_dirtree_without_files(src, dst):
   
      # getting the absolute path of the source
    # directory
    src = os.path.abspath(src)
     
    # making a variable having the index till which
    # src string has directory and a path separator
    src_prefix = len(src) + len(os.path.sep)
     
    if os.path.exists(dst):
        os.removedirs(dst) 
        # making the destination directory
        os.makedirs(dst)


     
    # doing os walk in source directory
    for root, dirs, files in os.walk(src):
        for dirname in dirs:
           
            # here dst has destination directory, 
            # root[src_prefix:] gives us relative
            # path from source directory 
            # and dirname has folder names
            dirpath = os.path.join(dst, root[src_prefix:], dirname)
             
            # making the path which we made by
            # joining all of the above three
            if len(files) > 0:
              print(dirpath + " - " + path_encode(dirpath) + " - " + str(len(files)))
            #os.mkdir(dirpath)
 

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


create_dirtree_without_files('/home/madhekar/work/home-media-app/data/raw-data/Madhekar/',
                             '/home/madhekar/temp/img_backup/ex1')
