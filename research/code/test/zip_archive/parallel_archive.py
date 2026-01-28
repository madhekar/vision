import subprocess
import os
from concurrent.futures import ProcessPoolExecutor

def zip_folder(folder_path, relative_path):
    """Zips a single folder using system subprocess (zip command)."""
    zip_name = os.path.join(relative_path, "archive", f"{os.path.basename(os.path.normpath(folder_path))}.zip")
    print(f"Starting compression: {folder_path} -> {zip_name}")
    
    # -r: recursive, -q: quiet
    cmd = ["zip", "-r", "-q", zip_name, folder_path]
    
    try:
        subprocess.run(cmd, cwd=relative_path, check=True, capture_output=True)
        print(f"Finished: {zip_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error zipping {folder_path}: {e}")

def main():
    # List of folders to zip
    folders_to_zip = ['home-media-app/models', 
                  'home-media-app/data/final-data/img',
                  'home-media-app/data/final-data/txt',
                  'home-media-app/data/final-data/video',
                  'home-media-app/data/final-data/audio',
                  'home-media-app/data/app-data']
    relative_path =['/home/madhekar/work'] * 6

    if not os.path.join(relative_path[0], 'archive'):
        os.makedirs(relative_path[0], 'archive')
        
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor() as executor:
        executor.map(zip_folder, folders_to_zip, relative_path)

if __name__ == "__main__":
    main()
