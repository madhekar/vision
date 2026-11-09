import os
import subprocess


def extract_exif_forall_image_subfolders(root_dir, gps_date_per_sub_folder):
    # This example extracts metadata into a tab-delimited file named 'exif_out.txt' inside each folder
    exif_command = ["exiftool", "-csv", "-n", "-r", "--ext", "csv", "-GPSLatitude", "-GPSLongitude", "-DateTimeOriginal"]

    # os.walk yields a 3-tuple (dirpath, dirnames, filenames) for each directory in the tree
    for dirpath, dirnames, filenames in os.walk(root_dir, gps_date_per_sub_folder):
        print(f"Processing folder: {dirpath}")

        if dirpath == root_dir:
            continue

        # Construct the execution command targeting the current subfolder
        # Example: exiftool [tags] "/path/to/subfolder"
        command = exif_command + [dirpath]
        
        try:
            # Run the command
            # if you want to redirect the output to a file per folder:
            with open(os.path.join(dirpath, "exif_out.csv"), "w") as outfile:
                subprocess.run(command, stdout=outfile, stderr=subprocess.PIPE, text=True, check=True)
                
        except subprocess.CalledProcessError as e:
            print(f"Error processing {dirpath}: {e.stderr}")

def extract_exif_forall_video_subfolders(root_dir, gps_date_per_sub_folder):
    # This example extracts metadata into a tab-delimited file named 'exif_out.txt' inside each folder
    exif_command = ["exiftool", "-csv", "-n", "-r", "--ext", "csv", "-GPSLatitude", "-GPSLongitude", "-CreateDate"]

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
            with open(os.path.join(dirpath, "exif_out.csv"), "w") as outfile:
                subprocess.run(command, stdout=outfile, stderr=subprocess.PIPE, text=True, check=True)
                
        except subprocess.CalledProcessError as e:
            print(f"Error processing {dirpath}: {e.stderr}")            


# Define the root directory and your target command
root_dir = "/home/madhekar/temp/filter/training"
gps_date_csv = "gps_date.csv"
extract_exif_forall_image_subfolders(root_dir=root_dir, gps_date_per_sub_folder=gps_date_csv)