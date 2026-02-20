from imagededup.methods import PHash
from imagededup.utils import plot_duplicates
import os

# Define the directory containing your images
image_directory = 'path/to/your/image/directory'

# 1. Initialize the hashing method
phasher = PHash()

# 2. Generate encodings for all images in the directory
# This creates a dictionary mapping filenames to their hashes
encodings = phasher.encode_images(image_dir=image_directory)

# 3. Find duplicates using the generated encodings
# The result is a dictionary mapping each image to a list of its duplicates
duplicates = phasher.find_duplicates(encoding_map=encodings)

# 4. (Optional) Plot duplicates for a specific image
# Replace 'your_image.jpg' with a filename from your directory to see its duplicates
if 'your_image.jpg' in duplicates:
    plot_duplicates(
        image_dir=image_directory,
        duplicate_map=duplicates,
        filename='your_image.jpg'
    )

# 5. (Optional) Print the duplicate pairs
print("Duplicate images found:")
for key, value in duplicates.items():
    if len(value) > 0:
        print(f"{key}: {value}")
