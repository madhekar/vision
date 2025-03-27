import os
import glob
import cv2
from utils.config_util import config
from utils.util import model_util as mu
from utils.util import statusmsg_util as sm

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

class Quality():
    def __init__(self, image_path, archivedir):
        self.image_path = image_path
        self.archivedir = archivedir

    def is_blurry(self, image, threshold=25.0):

        if image is not None:
            _image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            gray_image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)

            lap_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

            return lap_var < threshold  
        else:
            sm.add_messages('quality', 'e| Null image')  

    def is_valid_brisque_score(self, image, threshold = 50.0):

        if image:
            _image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
            
            brisque_score = cv2.quality.QualityBRISQUE_compute(
                gray,
                "/home/madhekar/work/home-media-app/models/brisque/brisque_model_live.yml",
                "/home/madhekar/work/home-media-app/models/brisque/brisque_range_live.yml",
            )    

            return brisque_score[0] > threshold
        else:
            sm.add_messages('quality', 'e| unable to load - NULL image') 

        return False   

    def find_quality_sharpness(self, image_sharpness_threshold, image_quality_threshold):
        """
        Find and Archive quality images
        """
        fnames = getRecursive(self.image_path)
        total_images = len(fnames)

        bad_quality_list = []

        print("Finding Good Quality Images ...!\n")

        sm.add_messages("quality", "s| Searching Good Quality Images...")

        for im in fnames:
            print(im[0], im[1])
            img = cv2.imread(os.path.join(im[0], im[1]))
            #is_b = self.is_blurry(img, image_sharpness_threshold)
            is_valid_brisque = self.is_valid_brisque_score(img, image_quality_threshold)

            # if is_b:
            #     quality_list.append(im)

            if is_valid_brisque:
                bad_quality_list.append(im)

        blurry_count = len(bad_quality_list)

        sm.add_messages('quality', f'w| {blurry_count} bad quality images found out off: {total_images}, percentage: {(blurry_count/ total_images) * 100}%')
                
        if len(bad_quality_list) != 0:
            a = input(f"Do you want to move/ archive these {len(bad_quality_list)} Images? Press Y or N:  ")

            space_saved = 0

            if a.strip().lower() == "y":
                for quality in bad_quality_list:
                    space_saved += os.path.getsize(os.path.join(quality[0], quality[1]))
                    #uuid_path = mu.create_uuid_from_string(quality[0]) 
                    uuid_path = mu.extract_subpath(self.image_path, quality[0])
                    if not os.path.exists(os.path.join(self.archivedir, uuid_path)):
                        os.makedirs(os.path.join(self.archivedir, uuid_path))
                    os.rename( os.path.join(quality[0], quality[1]), os.path.join(self.archivedir, uuid_path, quality[1]))
                    print(f"{quality} Moved Succesfully!")
                    #sm.add_messages("quality", f"s| file {quality} moved succesfully.")

                print(f"\n\nYou saved {round(space_saved / 1000000)} mb of Space!")
                sm.add_messages("quality",f"w| saved {round(space_saved / 1000000)} mb of Space.")
            else:
                print("Using quality Remover")
                sm.add_messages("quality", "s| No images are archived")
        else:
            print("No bad quality images Found :)")
            sm.add_messages("quality", "w| no bad quality images Found.")

    
def execute(source_name):
    (
        input_image_path,
        archive_quality_path,
        image_sharpness_threshold,
        image_quality_threshold
    ) = config.image_quality_config_load()

    input_image_path_updated = os.path.join(input_image_path,source_name)
    
    arc_folder_name = mu.get_foldername_by_datetime()  
    
    archive_quality_path = os.path.join(archive_quality_path, source_name, arc_folder_name)
  
    dr = Quality(image_path=input_image_path_updated, archivedir=archive_quality_path)

    dr.find_quality_sharpness(image_sharpness_threshold, image_quality_threshold)
    
if __name__ == "__main__":
    execute(source_name="")