from pathlib import Path

def count_files_pathlib(directory):
    count = len([f for f in Path(directory).rglob('*') if f.is_file()])
    return count

# Example usage
directory_path = '/mnt/zmdata/home-media-app/data/input-data/img/madhekar/'
file_count = count_files_pathlib(directory_path)
print(f"Total file count: {file_count}")