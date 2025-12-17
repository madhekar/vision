import os
from pathlib import Path

def get_folder_metrics(folder_path):
    """
    Calculates the total file count and size (in bytes) for a given folder path.

    Args:
        folder_path (str or Path): The path to the folder.

    Returns:
        tuple: (file_count, total_size_bytes)
    """
    file_count = 0
    total_size_bytes = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Skip if it is a symbolic link to avoid issues
            if not os.path.islink(fp):
                try:
                    total_size_bytes += os.path.getsize(fp)
                    file_count += 1
                except OSError:
                    # Handle cases where file might be inaccessible
                    print(f"Error accessing file: {fp}")
                    pass
    return file_count, total_size_bytes

def bytes_to_gb(bytes_size):
    """Converts bytes to gigabytes."""
    return bytes_size / (1024**3)

# Define the list of folders to process
folders_to_check = [
"/home/madhekar/work/home-media-app/data/input-data/img"
"/home/madhekar/work/home-media-app/data/input-data/video"
"/home/madhekar/work/home-media-app/data/input-data/txt"
"/home/madhekar/work/home-media-app/data/input-data/audio"
"/home/madhekar/work/home-media-app/data/input-data/error/img/duplicate"
"/home/madhekar/work/home-media-app/data/input-data/error/img/missing-data"
"/home/madhekar/work/home-media-app/data/input-data/error/img/quality"
"/home/madhekar/work/home-media-app/data/final-data/img"
"/home/madhekar/work/home-media-app/data/final-data/video"
"/home/madhekar/work/home-media-app/data/final-data/txt"
"/home/madhekar/work/home-media-app/data/final-data/audio"
]

print(f"{'Folder':<30} | {'File Count':<15} | {'Size (GB)':<15}")
print("-" * 64)

for folder in folders_to_check:
    if os.path.isdir(folder):
        count, size_bytes = get_folder_metrics(folder)
        size_gb = bytes_to_gb(size_bytes)
        print(f"{folder:<30} | {count:<15} | {size_gb:<15.4f}")
    else:
        print(f"'{folder}' is not a valid directory or does not exist.")

