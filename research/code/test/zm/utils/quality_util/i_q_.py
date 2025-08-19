import os
import glob
from PIL import Image
import pyiqa
import torch
from torchvision.transforms import ToTensor
from utils.config_util import config
from utils.util import model_util as mu
from utils.util import statusmsg_util as sm
from utils.util import storage_stat as ss

# /home/madhekar/work/home-media-app/data/input-data/img/AnjaliBackup/c5fdad7b-5b98-5890-956d-f0faef0f38bd/IMG_4367.PNG

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
    def __init__(self, image_path, archive_dir):
        self.image_path = image_path
        self.archive_dir = archive_dir

    def is_valid_size_and_score(self, img, metric, threshold=6.0):
      try:  
        if img is not None and os.path.getsize(img) > 512:
            im = Image.open(img).convert("RGB")
            h, w = im.size

            if w < 512 or h < 512:
                return False
            
            im_tensor = transform(im).unsqueeze(0).to(device)
            score = metric(im_tensor)
            fscore = score.item()

            print(f'{img} :: {h}:{w} :: {fscore}')

            return fscore < threshold
        else:
            sm.add_messages('quality', 'e| unable to load - NULL / Invalid image')
      except Exception as e:
                sm.add_messages("quality", f"e| error: {e} occurred while opening the image: {os.path.join(img[0], img[1])}")
                #os.remove(os.path.join(img[0], img[1]))
                #continue
      return False    
            
    """
    Find and Archive quality images
    """
    def find_images_quality(self, iqa_metric, image_quality_threshold):

        file_tuple_list = getRecursive(self.image_path)
        total_images = len(file_tuple_list)
        bad_quality_tuple_list = []
        
        sm.add_messages("quality", "s| Identifying Good Quality Images...")
        for im in file_tuple_list:
            img = os.path.join(im[0], im[1])
            if not self.is_valid_size_and_score(img, iqa_metric, image_quality_threshold):
                bad_quality_tuple_list.append(im)

        images_with_quality_issues_count = len(bad_quality_tuple_list)

        sm.add_messages("quality",f"w| {images_with_quality_issues_count} bad quality images found out off: {total_images}, percentage: {(images_with_quality_issues_count / total_images) * 100}%")
                
        if len(bad_quality_tuple_list) != 0:
            a = input(f"Do you want to move/ archive these {len(bad_quality_tuple_list)} Images? Press Y or N:  ")
            space_saved = 0

            if a.strip().lower() == "y":
                for quality in bad_quality_tuple_list:
                    space_saved += os.path.getsize(os.path.join(quality[0], quality[1]))
                    #uuid_path = mu.create_uuid_from_string(quality[0]) 
                    uuid_path = mu.extract_subpath(self.image_path, quality[0])
                    if not os.path.exists(os.path.join(self.archive_dir, uuid_path)):
                        os.makedirs(os.path.join(self.archive_dir, uuid_path))
                    os.rename( os.path.join(quality[0], quality[1]), os.path.join(self.archive_dir, uuid_path, quality[1]))
                    print(f"{quality} Moved Successfully!")
                    #sm.add_messages("quality", f"s| file {quality} moved successfully.")

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
        image_quality_threshold
    ) = config.image_quality_config_load()

    input_image_path_updated = os.path.join(input_image_path,source_name)
    
    arc_folder_name = mu.get_foldername_by_datetime()  
    
    archive_quality_path = os.path.join(archive_quality_path, source_name, arc_folder_name)
  
    dr = Quality(input_image_path_updated, archive_quality_path )

    iqa_metric = create_metric()

    dr.find_images_quality( iqa_metric, image_quality_threshold)

    ss.remove_empty_files_and_folders(input_image_path_updated)
    
if __name__ == "__main__":
    execute(source_name="")