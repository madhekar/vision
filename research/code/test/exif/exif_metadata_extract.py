import os
import subprocess

# Define the root directory and your target command
root_dir = "/mnt/zmdata/home-media-app/data/input-data/img/KIDDO/00a6c457-d3ba-54a2-a0a1-1643b185c177"
# This example extracts metadata into a tab-delimited file named 'exif_out.txt' inside each folder
exif_command = ["exiftool", "-csv", "-n", "-r", "-GPSLatitude", "-GPSLongitude", "-DateTimeOriginal"]

# os.walk yields a 3-tuple (dirpath, dirnames, filenames) for each directory in the tree
for dirpath, dirnames, filenames in os.walk(root_dir):
    print(f"Processing folder: {dirpath}")

    if dirpath == root_dir:
        continue
    
    # Construct the execution command targeting the current subfolder
    # Example: exiftool [tags] "/path/to/subfolder"
    command = exif_command + [dirpath]
    
    try:
        # Run the command
        # if you want to redirect the output to a file per folder:
        with open(os.path.join(dirpath, "exif_out.txt"), "w") as outfile:
            subprocess.run(command, stdout=outfile, stderr=subprocess.PIPE, text=True, check=True)
            
    except subprocess.CalledProcessError as e:
        print(f"Error processing {dirpath}: {e.stderr}")
