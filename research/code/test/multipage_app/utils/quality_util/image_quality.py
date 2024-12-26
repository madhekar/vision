
import os
import glob
import numpy as np
import cv2
from utils.config_util import config
from utils.util import model_util as mu
from utils.util import statusmsg_util as sm
import util

quality_threshold = 100

def getRecursive(rootDir):
    f_list = []
    for fn in glob.iglob(rootDir + "/**/*", recursive=True):
        if not os.path.isdir(os.path.abspath(fn)):
            f_list.append(
                (
                    str(os.path.abspath(fn)).replace(str(os.path.basename(fn)), ""),
                    os.path.basename(fn),
                )
            )
    return f_list

class Quality:
    def __init__(self, dirname, archivedir):
        self.dirname = dirname
        self.archivedir = archivedir

    def find_quality_sharpness(self, image_sharpness_threshold):
        """
        Find and Archive quality images
        """
        fnames = getRecursive(self.dirname)
        quality_list = []
        print("Finding quality Images Now!\n")
        sm.add_messages("quality", "Finding quality Images Now.")
        for image in fnames:
            with cv2.imread(os.path.join(image[0], image[1])) as image:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                lval = np.var(laplacian)
                if lval < image_sharpness_threshold:
                    quality_list.append(image)
        if len(quality_list) != 0:
            a = input("Do you want to move/ archive these {} Images? Press Y or N:  ".format(len(quality_list)))
            space_saved = 0
            if a.strip().lower() == "y":
                for quality in quality_list:
                    space_saved += os.path.getsize(os.path.join(quality[0], quality[1]))
                    if not os.path.exists(self.archivedir):
                        os.makedirs(self.archivedir)
                    uuid_path = mu.create_uuid_from_string(quality[0])    
                    os.rename( os.path.join(quality[0], quality[1]), os.path.join(self.archivedir, uuid_path, quality[1]))
                    print("{} Moved Succesfully!".format(quality))
                    sm.add_messages("quality", f"file {quality} moved succesfully.")

                print(f"\n\nYou saved {round(space_saved / 1000000)} mb of Space!")
                sm.add_messages("quality",f"\n\nYou saved {round(space_saved / 1000000)} mb of Space.")
            else:
                print("Using quality Remover")
                sm.add_messages("quality", "Using quality remover")
        else:
            print("No quality images Found :)")
            sm.add_messages("quality", "No quality images Found.")
   
    
def execute():
    input_image_path, archive_quality_path, image_sharpness_threshold = config.image_quality_config_load()
    arc_folder_name = util.get_foldername_by_datetime()  
    
    archive_quality_path = os.path.join(archive_quality_path, arc_folder_name)
  
    dr = Quality(dirname=input_image_path, archivedir=archive_quality_path)

    dr.find_quality_sharpness(image_sharpness_threshold)


if __name__ == "__main__":
    execute()