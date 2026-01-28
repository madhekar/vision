import os
import zipfile

def zip_multiple_folders(zip_filename, folder_paths):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder_path in folder_paths:
            # os.walk generates file names in a directory tree
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Create the relative path (arcname) to maintain folder structure in the zip
                    # os.path.relpath calculates the path relative to the current working directory
                    archive_name = os.path.relpath(file_path, os.getcwd())
                    zf.write(file_path, archive_name)

# Usage
zip_filename = 'z_media_archive.zip'
folders_to_zip = ['/home/madhekar/work/home-media-app/models', 
                  '/home/madhekar/work/home-media-app/data/final-data',
                  '/home/madhekar/work/home-media-app/data/app-data']
zip_multiple_folders(zip_filename, folders_to_zip)

print(f"Successfully created '{zip_filename}'")