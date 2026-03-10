from pathlib import Path

def count_files_pathlib(directory):
    # Use rglob('*') to find all files and directories recursively
    # Filter for files using is_file()
    # len() of the resulting list gives the count
    count = len([f for f in Path(directory).rglob('*') if f.is_file()])
    return count

# Example usage
directory_path = '.'
file_count = count_files_pathlib(directory_path)
print(f"Total file count: {file_count}")