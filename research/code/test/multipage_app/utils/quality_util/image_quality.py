
import os
import numpy as np
import cv2
from utils.config_util import config
import util


def sharpness_measure(img_path):
    image = cv2.imread(img_path)
    if image:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return  np.var(laplacian)
    else:
        return 0.0    
    
def execute():
    input_image_path, archive_quality_path = config.dedup_config_load()
    arc_folder_name = util.get_foldername_by_datetime()
    archive_dup_path = os.path.join(archive_quality_path, arc_folder_name)
    dr = DuplicateRemover(dirname=input_image_path, archivedir=archive_dup_path)
    dr.find_duplicates()


if __name__ == "__main__":
    execute()