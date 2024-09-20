from PIL import Image
import imagehash
import util
import os
import numpy as np
import yaml
import glob


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
    def __init__(self, dirname, archivedir,  hash_size=8):
        self.dirname = dirname
        self.archivedir = archivedir
        self.hash_size = hash_size

    def find_duplicates(self):
        """
        Find and Archive Duplicate images
        """
        fnames =  getRecursive(self.dirname)
        hashes = {}
        duplicates = []
        print("Finding Duplicate Images Now!\n")
        for image in fnames:
            with Image.open(os.path.join(image[0], image[1])) as img:
                temp_hash = imagehash.average_hash(img, self.hash_size)
                if temp_hash in hashes:
                    print("Duplicate {} \nfound for Image {}!\n".format(image, hashes[temp_hash]))
                    duplicates.append(image)
                else:
                    hashes[temp_hash] = image

        if len(duplicates) != 0:
            a = input("Do you want to delete these {} Images? Press Y or N:  ".format(len(duplicates)))
            space_saved = 0
            if a.strip().lower() == "y":
                for duplicate in duplicates:
                    space_saved += os.path.getsize( os.path.join(duplicate[0], duplicate[1])
                    )

                    os.rename(os.path.join(duplicate[0], duplicate[1]), os.path.join(self.archivedir, duplicate[1]))
                    print("{} Moved Succesfully!".format(duplicate))

                print(f"\n\nYou saved {round(space_saved / 1000000)} mb of Space!")
            else:
                print("Thank you for Using Duplicate Remover")
        else:
            print("WOW!! No Duplicates Found :)")

    def find_similar(self, location, similarity=80):
        fnames = os.listdir(self.dirname)
        threshold = 1 - similarity / 100
        diff_limit = int(threshold * (self.hash_size**2))

        with Image.open(location) as img:
            hash1 = imagehash.average_hash(img, self.hash_size).hash

        print("Finding Similar Images to {} Now!\n".format(location))
        for image in fnames:
            with Image.open(os.path.join(self.dirname, image)) as img:
                hash2 = imagehash.average_hash(img, self.hash_size).hash

                if np.count_nonzero(hash1 != hash2) <= diff_limit:
                    print(
                        "{} image found {}% similar to {}".format(
                            image, similarity, location
                        )
                    )

if __name__=='__main__':
    with open("dup_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        print("**** duplicate archiver properties****")
        print(dict)
        print("**************************************")

        input_image_path = dict["duplicate"]["input_image_path"]
        archive_dup_path = dict["duplicate"]["archive_dup_path"]
        arc_folder_name = util.get_foldername_by_datetime()
        
        archive_dup_path = os.path.join(archive_dup_path, arc_folder_name)

    dr = DuplicateRemover(
        dirname=input_image_path,
        archivedir=archive_dup_path
    )

    dr.find_duplicates()