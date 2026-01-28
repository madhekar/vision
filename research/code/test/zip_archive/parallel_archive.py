import subprocess
import os
from concurrent.futures import ProcessPoolExecutor

def zip_folder(folder_path):
    """Zips a single folder using system subprocess (zip command)."""
    zip_name = f"{os.path.basename(os.path.normpath(folder_path))}.zip"
    print(f"Starting compression: {folder_path} -> {zip_name}")
    
    # -r: recursive, -q: quiet
    cmd = ["zip", "-r", "-q", zip_name, folder_path]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Finished: {zip_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error zipping {folder_path}: {e}")

def main():
    # List of folders to zip
    folders_to_zip = ['/home/madhekar/work/home-media-app/models', 
                  '/home/madhekar/work/home-media-app/data/final-data/img',
                  '/home/madhekar/work/home-media-app/data/final-data/txt',
                  '/home/madhekar/work/home-media-app/data/final-data/video',
                  '/home/madhekar/work/home-media-app/data/final-data/audio',
                  '/home/madhekar/work/home-media-app/data/app-data']
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor() as executor:
        executor.map(zip_folder, folders_to_zip)

if __name__ == "__main__":
    main()
