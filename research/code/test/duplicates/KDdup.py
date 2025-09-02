   
import ImgToHash as ah
import find
import random
from helper import build_tree, save_results
"""
$ deduplication search \
--images-path datasets/potatoes \
--output-path outputs \
--tree-type KDTree \
--threshold 40 \
--parallel f \
--nearest-neighbors 5 \
--hash-algorithm phash \
--hash-size 8 \
--distance-metric manhattan \
--query datasets/potatoes/2018-12-11-15-031193.png

"""
# images_path = '/home/madhekar/work/home-media-app/data/input-data/img/madhekar'#'/home/madhekar/work/home-media-app/data/input-data/img/madhekar/cc2af672-0277-5d32-ad7d-2dfac3662e7b'#'/home/madhekar/work/home-media-app/data/input-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99'#'/home/madhekar/temp/faces'
# output_path = './result'
# hash_algo = 'phash'
# hash_size = 8
# tree_type = 'KDTree'
# distance_metric = 'manhattan'
# nearest_neighbors = 5
# leaf_size = 15
# parallel = 'f'
# batch_size = 32
# threshold = 5
# image_w = 512
# image_h = 512
# query = "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/767bfd11-78fa-573e-ac47-cb74883dc6c9/1.jpg"#'/home/madhekar/work/home-media-app/data/input-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/IMG_5942.PNG'#'/home/madhekar/temp/faces/Esha/CIMG0981.JPG'

# df_dataset, _ = ah.ImageToHash(images_path, hash_size=hash_size, hash_algo=hash_algo).build_dataset(parallel=parallel, batch_size=batch_size)

# print(df_dataset)

# x = find.search(df_dataset, output_path, tree_type, distance_metric, nearest_neighbors, leaf_size, parallel, batch_size,threshold, image_w, image_h, query)

# print(x)


'''----'''

img_file_list='/home/madhekar/work/home-media-app/data/input-data/img/madhekar'
output_path='./result'
hash_size=8
tree_type='KDTree'
distance_metric='manhattan'
nearest_neighbors=5
hash_algo = 'phash'
leaf_size=15
parallel='y'
batch_size=64
threshold=5
backup_keep='y'
backup_duplicate='y' 
safe_deletion='y'
image_h = 512
image_w = 512


# def delete(df_dataset, img_file_list, output_path, hash_size, tree_type, distance_metric, nearest_neighbors,
#            leaf_size, parallel, batch_size, threshold, backup_keep, backup_duplicate, safe_deletion, image_w=128,
#            image_h=128):
    # Build the tree
df_dataset, _ = ah.ImageToHash(img_file_list, hash_size=hash_size, hash_algo=hash_algo).build_dataset(parallel=parallel, batch_size=batch_size)
near_duplicate_image_finder = build_tree(df_dataset, tree_type, distance_metric, leaf_size, parallel, batch_size)

# Find duplicates
to_keep, to_remove, dict_image_to_duplicates = near_duplicate_image_finder.find_all_near_duplicates(
    nearest_neighbors,
    threshold)

print('We have found {0}/{1} duplicates in folder'.format(len(to_remove), len(img_file_list)))

# Show a duplicate
if len(dict_image_to_duplicates) > 0:
    random_img = random.choice(list(dict_image_to_duplicates.keys()))
    near_duplicate_image_finder.show_an_image_duplicates(dict_image_to_duplicates, random_img, output_path,
                                                            image_w=image_w,
                                                            image_h=image_h)
# Save results
save_results(to_keep, to_remove, hash_size, threshold, output_path, backup_keep, backup_duplicate, safe_deletion)
print(to_keep, to_remove)