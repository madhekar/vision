from PIL import Image
import imagehash
import os

def generate_hashes(image_dir):
    hashes = {}
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, filename)
            try:
                # Using pHash with a hash size (e.g., 8x8 gives a 64-bit hash)
                current_hash = imagehash.phash(Image.open(img_path))
                hashes[filename] = current_hash
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
    return hashes

image_hashes = generate_hashes("your_image_directory")
# Store hashes as integers for better performance/compatibility with some libraries
hash_list = [(int(h.format(sep=''), 16), fname) for fname, h in image_hashes.items()]
# Or just the hashes as ImageHash objects, depending on the library
hash_objects = list(image_hashes.values())


import vptree
import numpy as np
from PIL import Image
import imagehash

# The distance function for ImageHash objects is the built-in difference
def hamming_distance_func(hash1, hash2):
    return hash1 - hash2 # ImageHash objects overload the minus operator for hamming distance

# Build the VP tree
# You would use 'hash_objects' from the previous step
tree = vptree.VPTree(hash_objects, hamming_distance_func)

# Query the tree
query_image_path = "query_image.jpg"
query_hash = imagehash.phash(Image.open(query_image_path))

# Find the nearest neighbors (e.g., 10 nearest neighbors)
# result format: (distance, hash_object)
neighbors = tree.get_n_nearest_neighbors(query_hash, 10)

# Find all images within a certain distance threshold (e.g., distance < 5)
# result format: (distance, hash_object)
nearby_images = tree.get_all_in_range(query_hash, 5)

for distance, hash_obj in nearby_images:
    # You will need to map the hash_obj back to the filename in a real application
    # (e.g., by storing tuples of (hash, filename) in the vptree data)
    print(f"Found image with Hamming distance: {distance}")
