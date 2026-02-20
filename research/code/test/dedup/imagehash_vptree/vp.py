import random
from PIL import Image
from collections import defaultdict
import imagehash
import os

'''
[
['/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/52d3b8f6-b993-50bf-b2a0-0debd314f98a/IMG_8133.jpeg', 
'/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/52d3b8f6-b993-50bf-b2a0-0debd314f98a/IMG_8132.jpeg'], 

['/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/52d3b8f6-b993-50bf-b2a0-0debd314f98a/IMG_8134.jpeg', 
'/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/52d3b8f6-b993-50bf-b2a0-0debd314f98a/IMG_8132.jpeg'], 

['/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/52d3b8f6-b993-50bf-b2a0-0debd314f98a/IMG_8132.jpeg', 
'/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/52d3b8f6-b993-50bf-b2a0-0debd314f98a/IMG_8133.jpeg', 
'/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/52d3b8f6-b993-50bf-b2a0-0debd314f98a/IMG_8134.jpeg'], 

['/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/8d9e4356-64fa-5ae4-bb63-3614752f08fa/IMG_8106.jpeg', 
'/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/8d9e4356-64fa-5ae4-bb63-3614752f08fa/IMG_8109.jpeg'], 

['/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/8d9e4356-64fa-5ae4-bb63-3614752f08fa/IMG_8109.jpeg', 
'/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/8d9e4356-64fa-5ae4-bb63-3614752f08fa/IMG_8106.jpeg'], 

['/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/c8bf737f-6bcb-5a84-b13c-d588f324788e/IMG_5403.jpeg', 
'/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/c8bf737f-6bcb-5a84-b13c-d588f324788e/IMG_5404.jpeg'], 

['/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/c8bf737f-6bcb-5a84-b13c-d588f324788e/IMG_5404.jpeg', 
'/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/c8bf737f-6bcb-5a84-b13c-d588f324788e/IMG_5403.jpeg'], 

['/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/bb617fd9-0341-5ce0-aaaa-32b99451d302/image1-23.jpg', 
'/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/bb617fd9-0341-5ce0-aaaa-32b99451d302/image1-small-24.jpg'], 

['/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/bb617fd9-0341-5ce0-aaaa-32b99451d302/image1-small-24.jpg', 
'/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/bb617fd9-0341-5ce0-aaaa-32b99451d302/image1-23.jpg'], 

['/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/baa04b52-1d78-55c5-b79d-71905e7e40aa/IMG_6156.jpeg', 
'/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/baa04b52-1d78-55c5-b79d-71905e7e40aa/IMG_6157.jpeg'], 

['/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/baa04b52-1d78-55c5-b79d-71905e7e40aa/IMG_6157.jpeg', 
'/mnt/zmdata/home-media-app/data/input-data/img/Berkeley/baa04b52-1d78-55c5-b79d-71905e7e40aa/IMG_6156.jpeg']
]

'''



def generate_hashes(image_dir):
    hashes = {}
    file_paths = [os.path.join(root, name) for root, dirs, files in os.walk(image_dir) for name in files]
    for img_path in  file_paths: #os.listdir(image_dir):
        # if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        #     img_path = os.path.join(image_dir, filename)
            try:
                # Using pHash with a hash size (e.g., 8x8 gives a 64-bit hash)
                current_hash = imagehash.phash(Image.open(img_path))
                #print(current_hash)
                hashes[img_path] = current_hash
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
    return hashes

# image_hashes = generate_hashes("your_image_directory")
# # Store hashes as integers for better performance/compatibility with some libraries
# hash_list = [(int(h.format(sep=''), 16), fname) for fname, h in image_hashes.items()]
# # Or just the hashes as ImageHash objects, depending on the library
# hash_objects = list(image_hashes.values())


# Assume a library function to calculate distance, e.g., Hamming distance for hashes
def hamming_distance(hash1, hash2):
    
    # print(f"Type of hash1: {type(hash1)}{hash1}")
    # print(f"Type of hash2: {type(hash2)}{hash2}")
    if hash1 is None or hash2 is None:
        print("Error: One of the images could not be hashed properly.")
    else:
        # The ^ operator works correctly on two valid ImageHash objects
        hamming_distance = hash1 - hash2
        #print(f"Hamming distance: {hamming_distance}")
    # Pythonic way to calculate Hamming distance (count of differing bits)
    return  hamming_distance #bin(hash1 ^ hash2).count('1')

# Conceptual class - actual implementation will be in a specific library
class VPTree:
    def __init__(self, items, idict, distance_function):
        # ... implementation details ...
        self.items = items.values()
        self.idict = idict
        self.distance_function = distance_function

    def find_duplicates(self, threshold):
        #duplicates = defaultdict(list)
        res = []
        for i, item in enumerate(self.items):
            #print(i, item)
            # Query the tree for neighbors within the threshold
            # The query_tree method is the core VP-tree functionality
            neighbors = self.query_tree_dict(i, item, threshold) 
            
            if len(neighbors) > 1:
                res.append(neighbors)
            #   print(f'neighbors: {neighbors}')
            #   for neighbor_item in neighbors:
            #     #print(f"---> {distance} - {neighbor_item}")
            #     # Add to duplicates list, ensuring we don't count an item as its own duplicate
            #     if neighbor_item != self.idict[item]:
            #         duplicates[item].append(neighbor_item) # Use tuple for unique pairs
        return res
    
    def query_tree(self, query_item, threshold):
        # ... conceptual nearest neighbor search implementation ...
        # In a real library, this would efficiently prune the search space
        #print(f"->>>>{query_item}")
        print(f"H {query_item}")
        found = []
        for item in self.items:
            if item != query_item:
                dist = self.distance_function(query_item, item)
                if dist <= threshold:
                    found.append((dist, item))
                    print(f"C {item} -- {found}")
        return found
    
    def query_tree_dict(self, i, query_item, threshold):
        found = []#defaultdict(list)
        found.append(self.idict[query_item])
        for item in self.items:
            if item != query_item:
                #print(f"===={query_item}")
                dist = self.distance_function(query_item, item)
                if dist <= threshold:
                    #print(f"{i}:{self.idict[query_item]}=={self.idict[item]}")
                    #found[self.idict[query_item]].append(self.idict[item])
                    found.append(self.idict[item])
        #print(found)            
        return found 

# Example Usage:
# Assume these are 64-bit integer hashes for images
# image_hashes = [
#     1234567890123456, 
#     1234567890123457, # Near duplicate of the first (distance 1)
#     9876543210987654, 
#     1234567890123456, # Exact duplicate
#     5555555555555555
# ]
image_dir = "/mnt/zmdata/home-media-app/data/input-data/img/madhekar"
image_hashes = generate_hashes(image_dir=image_dir)
idict = {v:k for k, v in image_hashes.items()}
# Build the VP Tree (using the conceptual class)
# A real library would have a 'build' method, e.g., `tree = vptree.VPTree(image_hashes, hamming_distance)`
vp_tree = VPTree(image_hashes, idict, hamming_distance)
#print('----', vp_tree)
# Find duplicates with a threshold of 1 (exact or 1-bit difference)
# This will return pairs of items that are considered duplicates
duplicate_pairs = vp_tree.find_duplicates(threshold=2)

print("Duplicate pairs found:", duplicate_pairs)
