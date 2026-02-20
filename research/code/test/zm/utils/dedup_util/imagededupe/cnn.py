import os
from utils.config_util import config
from utils.util import statusmsg_util as sm
from utils.util import storage_stat as ss
from utils.util import model_util  as mu
from imagededup.methods import CNN
from tqdm import tqdm

"""
Archive duplicate images
"""
def archive_images(image_path, archive_path, duplicate_filter_img_list):                
    if len(duplicate_filter_img_list) != 0:
        space_saved = 0
        image_cnt =0 
        for duplicate in tqdm(duplicate_filter_img_list, total=len(duplicate_filter_img_list), desc='duplicate filter files'):
                duplicate = os.path.join(image_path, duplicate)
                space_saved += os.path.getsize(duplicate)
                image_cnt += 1
                #uuid_path = mu.create_uuid_from_string(duplicate[0]) 
                uuid_path = mu.extract_subpath(image_path, os.path.dirname(duplicate))
                print(uuid_path)
                if not os.path.exists(os.path.join(archive_path, uuid_path)):
                    os.makedirs(os.path.join(archive_path, uuid_path))
                os.rename( duplicate, os.path.join(archive_path, uuid_path, os.path.basename(duplicate)))
                print(f"{duplicate} Moved Successfully!")
                sm.add_messages("duplicate", f"s| file {duplicate} moved successfully.")

        print(f"\n\n saved {round(space_saved / 1000000)} mb of Space, {image_cnt} images archived.")
        sm.add_messages("duplicate",f"s| saved {round(space_saved / 1000000)} mb of Space, {image_cnt} images archived.")
    else:
        print("no duplicate or filtered images Found.:)")
        sm.add_messages("duplicate", "s| no duplicate or filtered images Found.")

def remove_duplicates(input_image_path, archive_duplicates_path):

    # 1. Initialize the CNN encoder
    cnn_encoder = CNN()

    # 2. Find duplicates directly (encodings are generated internally)
    # min_similarity_threshold can be adjusted (e.g., 0.9 for high similarity)
    duplicates = cnn_encoder.find_duplicates(
        image_dir=input_image_path,
        min_similarity_threshold=0.90, # Adjust as needed
        scores=False, # Set to True to get similarity scores
        recursive=True
    )
    dup_images_to_remove = []
    # The 'duplicates' dictionary will have filenames as keys and a list of tuples (duplicate_filename, score) as values
    print("Duplicate images and their similarity scores found with CNN:")
    for key, value in duplicates.items():
        if len(value) > 0:
            print(f"{key}: {value}")
            res = [(key, v) for v in value]
            dup_images_to_remove.extend(res)      
    unique_combination_set = set(tuple(sorted(c)) for c in dup_images_to_remove)     
    to_remove = [e[1] for e in unique_combination_set]    
    if len(to_remove) > 0:
        archive_images(image_path=input_image_path, archive_path=archive_duplicates_path, duplicate_filter_img_list=to_remove)


def execute(source_name):
    input_image_path, archive_dup_path = config.dedup_config_load()
    input_image_path = os.path.join(input_image_path, source_name)
    arc_folder_name_dt = mu.get_foldername_by_datetime()
    archive_dup_path_update = os.path.join(
        archive_dup_path, source_name, arc_folder_name_dt
    )
    sm.add_messages("duplicate", f"s| Images input Folder Path: {input_image_path}")
    sm.add_messages("duplicate", f"s| Images archive folder path: {archive_dup_path_update}")
    ss.create_folder(archive_dup_path_update)

    #file_paths = [os.path.join(root, name) for root, dirs, files in os.walk(input_image_path) for name in files]
    remove_duplicates(input_image_path, archive_dup_path_update)

    return "success"

if __name__=='__main__':
    img_file_list='/home/madhekar/work/home-media-app/data/input-data/img/madhekar'
    output_path='./result'
    to_keep, to_remove = remove_duplicates(img_file_list, output_path)    

    print(to_remove)