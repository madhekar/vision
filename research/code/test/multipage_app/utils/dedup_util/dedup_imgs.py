from PIL import Image
import imagehash
#from utils import util
import os
import numpy as np
import glob
from utils.config_util import config
from utils.util import model_util as mu
from utils.util import statusmsg_util as sm

def getRecursive(rootDir):
    f_list = []
    for fn in glob.iglob(rootDir + "/**/*", recursive=True):
        if not os.path.isdir(os.path.abspath(fn)):
            f_list.append((str(os.path.abspath(fn)).replace(str(os.path.basename(fn)),''), os.path.basename(fn)))
    return f_list

def getRecursive_by_type(rootDir, types):
    f_list = []
    for t in types:
        for fn in glob.iglob(rootDir + "/**/" + t, recursive=True):
            if not os.path.isdir(os.path.abspath(fn)):
                f_list.append(
                    (
                        str(os.path.abspath(fn)).replace(str(os.path.basename(fn)), ""),
                        os.path.basename(fn),
                    )
                )
    return f_list

class DuplicateRemover:
    def __init__(self, image_path, archivedir,   hash_size=8):
        self.image_path = image_path
        self.archivedir = archivedir
        self.hash_size = hash_size

    def find_duplicates(self):
        """
        Find and Archive Duplicate images
        """
        fnames =  getRecursive(self.image_path)
        hashes = {}
        duplicates = []
        print("Finding Duplicate Images Now!\n")
        sm.add_messages("duplicate","s| Searching... Duplicate Images Now.")
        for image in fnames:
            try:
              with Image.open(os.path.join(image[0], image[1])) as img:
                temp_hash = imagehash.average_hash(img, self.hash_size)
                if temp_hash in hashes:
                    print(f"Duplicate {image} \nfound for Image {hashes[temp_hash]}!\n")
                    sm.add_messages("duplicate", f"w|Duplicate {image} found for Image {hashes[temp_hash]}")
                    duplicates.append(image)
                else:
                    hashes[temp_hash] = image
            except(IOError) as e:
                sm.add_messages("duplicate", f"e| error: {e} ocurred while opening the image: {os.path.join(image[0], image[1])}")
                os.remove(os.path.join(image[0], image[1]))
                continue

        if len(duplicates) != 0:
            a = input(f"w| Do you want to move/ archive these {str(len(duplicates))} Images? Press Y or N:")
            space_saved = 0
            if a.strip().lower() == "y":
                for duplicate in duplicates:
                    space_saved += os.path.getsize( os.path.join(duplicate[0], duplicate[1]))
                    if not os.path.exists(self.archivedir):
                        os.makedirs(self.archivedir)
                    #uuid_path = mu.create_uuid_from_string(duplicate[0]) # ? use old uuid already generated
                    uuid_path = mu.extract_subpath(self.image_path, duplicate[0])
                    if not os.path.exists(os.path.join(self.archivedir, uuid_path)):
                        os.makedirs(os.path.join(self.archivedir, uuid_path)) 
                    os.rename(os.path.join(duplicate[0], duplicate[1]), os.path.join(self.archivedir, uuid_path, duplicate[1]))
                    print(f"{duplicate} Moved Succesfully!")
                    sm.add_messages("duplicate", f"s| {duplicate} Moved Succesfully!")
                print(f"\n\nYou saved {round(space_saved / 1000000)} mb of Space!")
                sm.add_messages("duplicate", f"s| saved {round(space_saved / 1000000)} mb of Space!")
            else:
                print("Using Duplicate Remover")
                sm.add_messages("s| duplicate", "Using Duplicate Remover.")
        else:
            print("No Duplicate images Found :)")
            sm.add_messages("w| duplicate", "No Duplicate images Found.")

    def find_similar(self, location, similarity=80):
        fnames = os.listdir(self.image_path)
        threshold = 1 - similarity / 100
        diff_limit = int(threshold * (self.hash_size**2))

        with Image.open(location) as img:
            hash1 = imagehash.average_hash(img, self.hash_size).hash

        sm.add_messages("duplicate", f"s| Searching... Similar Images in {location}.")
        for image in fnames:
            with Image.open(os.path.join(self.image_path, image)) as img:
                hash2 = imagehash.average_hash(img, self.hash_size).hash

                if np.count_nonzero(hash1 != hash2) <= diff_limit:
                    print("{} image found {}% similar to {}".format(image, similarity, location))
                    sm.add_messages("duplicate", f"w| {image} image /w similarity score: {similarity}% found in: {location}")

def execute(source_name):
       input_image_path, archive_dup_path = config.dedup_config_load()
       arc_folder_name_dt = mu.get_foldername_by_datetime()
       archive_dup_path_update = os.path.join(archive_dup_path, source_name, arc_folder_name_dt)
       dr = DuplicateRemover( dirname=input_image_path,  archivedir=archive_dup_path_update)
       dr.find_duplicates()                

if __name__=='__main__':
    execute()