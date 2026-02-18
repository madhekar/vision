   
import os
from utils.dedup_util.phash import ImgToHash as ah
from utils.dedup_util.phash import search_similar as search_similar
from utils.config_util import config
from utils.util import statusmsg_util as sm
from utils.util import storage_stat as ss
from utils.util import model_util  as mu
import random
from utils.dedup_util.phash import helper_functions as hf #import build_tree, save_results

def remove_duplicates(img_file_list, output_path, hash_size=256, tree_type='KDTree', distance_metric='manhattan', nearest_neighbors=5,
           leaf_size=16, hash_algo='phash', parallel='y', batch_size=64, threshold=5, backup_keep=False, backup_duplicate=True, safe_deletion=False, image_w=32, image_h=32):
    
    # Build the tree
    df_dataset, _ = ah.ImageToHash(img_file_list, hash_size=hash_size, hash_algo=hash_algo).build_dataset(parallel=parallel, batch_size=batch_size)
    near_duplicate_image_finder = hf.build_tree(df_dataset, tree_type, distance_metric, leaf_size, parallel, batch_size)

    # Find duplicates
    to_keep, to_remove, dict_image_to_duplicates = near_duplicate_image_finder.find_all_near_duplicates(nearest_neighbors,threshold)

    print('We have found {0}/{1} duplicates in folder'.format(len(to_remove), len(img_file_list)))
    sm.add_messages("duplicate", f"w| found {len(to_remove)}/{len(img_file_list)} duplicates in folder")

    # Show a duplicate
    # if len(dict_image_to_duplicates) > 0:
    #     random_img = random.choice(list(dict_image_to_duplicates.keys()))
    #     near_duplicate_image_finder.show_an_image_duplicates(dict_image_to_duplicates, random_img, output_path, image_w=image_w, image_h=image_h)

    # Save results
    hf.save_results(to_keep, to_remove, hash_size, threshold, output_path, backup_keep, backup_duplicate, safe_deletion)

    return to_keep, to_remove


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

    remove_duplicates(input_image_path, archive_dup_path_update)

if __name__=='__main__':
    img_file_list='/home/madhekar/work/home-media-app/data/input-data/img/madhekar'
    output_path='./result'
    to_keep, to_remove = remove_duplicates(img_file_list, output_path)    

    print(to_remove)