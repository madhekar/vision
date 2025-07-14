import subprocess
import os

def find_duplicate_images(directory):
    """Finds duplicate images in a directory using fdupes."""

    try:
        result = subprocess.run(['fdupes', '-l', '-S', '-r', directory], capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        if not output:
            print("No duplicate images found.")
            return

        duplicate_groups = output.split('\n\n')
        duplicates = []
        for group in duplicate_groups:
            files = group.split('\n')
            duplicates.append(files)

        for group in duplicates:
            print("Duplicate group:")
            for file_path in group:
                print(f"  - {file_path}")

    except FileNotFoundError:
        print("fdupes not found. Please install it (e.g., `sudo apt-get install fdupes` on Debian/Ubuntu).")
    except subprocess.CalledProcessError as e:
        print(f"Error running fdupes: {e}")


if __name__ == "__main__":
    target_directory = "/home/madhekar/work/home-media-app/data/input-data-1/error/img/quality/AnjaliBackup/20250328-113554" # Replace with the actual directory
    find_duplicate_images(target_directory)