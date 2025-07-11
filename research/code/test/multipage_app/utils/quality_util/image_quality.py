import os
import glob
import cv2
from PIL import Image
import pyiqa
import torch
from torchvision.transforms import ToTensor
from utils.config_util import config
from utils.util import model_util as mu
from utils.util import statusmsg_util as sm
from utils.util import storage_stat as ss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = ToTensor()

def create_metric():
    print(pyiqa.list_models())
    iqa_metric = pyiqa.create_metric("niqe", device=device)
    return iqa_metric

def getRecursive(rootDir):
    f_list = []
    for fn in glob.iglob(rootDir + "/**/*", recursive=True):
        if not os.path.isdir(os.path.abspath(fn)):
            f_list.append(( str(os.path.abspath(fn)).replace(str(os.path.basename(fn)), ""), os.path.basename(fn)))
    return f_list

class Quality():
    def __init__(self, image_path, archivedir):
        self.image_path = image_path
        self.archivedir = archivedir
        # self.brisque_model_path = brisque_model_path
        # self.model_live_file = model_live_file
        # self.model_range_file = model_range_file

    # def is_blurry(self, image, threshold=25.0):

    #     if image is not None:
    #         _image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #         gray_image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)

    #         lap_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    #         return lap_var < threshold  
    #     else:
    #         sm.add_messages('quality', 'e| Null image')  

    # def is_valid_brisque_score(self, image, threshold = 6.0):

    #     if image is not None:
    #         h, w, _ = image.shape
    #         if w < 512 or h < 512:
    #             return False

    #         _image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #         gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
            
    #         brisque_score = cv2.quality.QualityBRISQUE_compute(
    #             gray,
    #             os.path.join(self.brisque_model_path, self.model_live_file),
    #             os.path.join(self.brisque_model_path, self.model_range_file)
    #         )    

    #         return brisque_score[0] < threshold
    #     else:
    #          

    #     return False   

    def is_valid_size_and_score(self, img, metric, threshold=6.0):
        if img is not None:
            img = Image.open(img).convert("RGB")
            h, w, _ = img.shape
            if w < 512 or h < 512:
                return False
            
            img_tensor = transform(img).unsqueeze(0).to(device)
            score = metric(img_tensor)
            fscore = score.item()

            return fscore < threshold
        else:
            sm.add_messages('quality', 'e| unable to load - NULL image')
        return False    
            
    """
    Find and Archive quality images
    """
    def find_images_quality(self, iqa_metric, image_quality_threshold):

        fnames = getRecursive(self.image_path)
        total_images = len(fnames)
        bad_quality_list = []
        sm.add_messages("quality", "s| Searching Good Quality Images...")

        for im in fnames:
            print(im[0], im[1])
            img = cv2.imread(os.path.join(im[0], im[1]))
            if not self.is_valid_size_and_score(img, iqa_metric, image_quality_threshold):
                bad_quality_list.append(im)

        images_with_quality_issues_count = len(bad_quality_list)

        sm.add_messages(
            "quality",
            f"w| {images_with_quality_issues_count} bad quality images found out off: {total_images}, percentage: {(images_with_quality_issues_count / total_images) * 100}%"
        )
                
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

    
    """
        input_image_path,
        archive_quality_path,
        image_quality_threshold,

    """
def execute(source_name):
    (
        input_image_path,
        archive_quality_path,
        #image_sharpness_threshold,
        image_quality_threshold,
        # brisque_model_config_path,
        # brisque_model_live_file,
        # brisque_range_live_file
    ) = config.image_quality_config_load()

    input_image_path_updated = os.path.join(input_image_path,source_name)
    
    arc_folder_name = mu.get_foldername_by_datetime()  
    
    archive_quality_path = os.path.join(archive_quality_path, source_name, arc_folder_name)
  
    dr = Quality(input_image_path_updated, archive_quality_path )

    iqa_metric = create_metric()

    dr.find_images_quality_( iqa_metric, image_quality_threshold)

    ss.remove_empty_folders(input_image_path_updated)
    
if __name__ == "__main__":
    execute(source_name="")