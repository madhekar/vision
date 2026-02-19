import random
from PIL import Image
import imagehash
import os

def generate_hashes(image_dir):
    hashes = {}
    file_paths = [os.path.join(root, name) for root, dirs, files in os.walk(image_dir) for name in files]
    for filename in  file_paths: #os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, filename)
            try:
                # Using pHash with a hash size (e.g., 8x8 gives a 64-bit hash)
                current_hash = imagehash.phash(Image.open(img_path))
                hashes[filename] = current_hash
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
    return hashes

# image_hashes = generate_hashes("your_image_directory")
# # Store hashes as integers for better performance/compatibility with some libraries
# hash_list = [(int(h.format(sep=''), 16), fname) for fname, h in image_hashes.items()]
# # Or just the hashes as ImageHash objects, depending on the library
# hash_objects = list(image_hashes.values())


# Assume a library function to calculate distance, e.g., Hamming distance for hashes
def hamming_distance(hash1, hash2):
    # Pythonic way to calculate Hamming distance (count of differing bits)
    return bin(hash1 ^ hash2).count('1')

# Conceptual class - actual implementation will be in a specific library
class VPTree:
    def __init__(self, items, distance_function):
        # ... implementation details ...
        self.items = items
        self.distance_function = distance_function

    def find_duplicates(self, threshold):
        duplicates = set()
        for i, item in enumerate(self.items):
            # Query the tree for neighbors within the threshold
            # The query_tree method is the core VP-tree functionality
            neighbors = self.query_tree(item, threshold) 
            for distance, neighbor_item in neighbors:
                # Add to duplicates list, ensuring we don't count an item as its own duplicate
                if neighbor_item != item:
                    duplicates.add(tuple(sorted((item, neighbor_item)))) # Use tuple for unique pairs
        return duplicates
    
    def query_tree(self, query_item, threshold):
        # ... conceptual nearest neighbor search implementation ...
        # In a real library, this would efficiently prune the search space
        found = []
        for item in self.items:
            if item != query_item:
                dist = self.distance_function(query_item, item)
                if dist <= threshold:
                    found.append((dist, item))
        return found

# Example Usage:
# Assume these are 64-bit integer hashes for images
image_hashes = [
    1234567890123456, 
    1234567890123457, # Near duplicate of the first (distance 1)
    9876543210987654, 
    1234567890123456, # Exact duplicate
    5555555555555555
]

# Build the VP Tree (using the conceptual class)
# A real library would have a 'build' method, e.g., `tree = vptree.VPTree(image_hashes, hamming_distance)`
vp_tree = VPTree(image_hashes, hamming_distance)

# Find duplicates with a threshold of 1 (exact or 1-bit difference)
# This will return pairs of items that are considered duplicates
duplicate_pairs = vp_tree.find_duplicates(threshold=1)

print("Duplicate pairs found:", duplicate_pairs)
