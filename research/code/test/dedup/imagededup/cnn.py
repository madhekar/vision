from imagededup.methods import CNN

# Define the directory containing your images
image_directory = "/mnt/zmdata/home-media-app/data/input-data/img/madhekar"

# 1. Initialize the CNN encoder
cnn_encoder = CNN()

# 2. Find duplicates directly (encodings are generated internally)
# min_similarity_threshold can be adjusted (e.g., 0.9 for high similarity)
duplicates = cnn_encoder.find_duplicates(
    image_dir=image_directory,
    min_similarity_threshold=0.95, # Adjust as needed
    scores=False, # Set to True to get similarity scores
    recursive=True
)

# The 'duplicates' dictionary will have filenames as keys and a list of tuples (duplicate_filename, score) as values
print("Duplicate images and their similarity scores found with CNN:")
for key, value in duplicates.items():
    if len(value) > 0:
        print(f"{key}: {value}")
