import os
from datetime import datetime
from pathlib import Path
from PIL import Image, ExifTags

# Supported image extensions
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tiff', '.webp'}

def get_image_date(image_path: Path) -> datetime:
    """Extracts EXIF DateTimeOriginal, falling back to file modification time."""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if exif:
                # 36867 is DateTimeOriginal, 306 is DateTime
                for tag_id in (36867, 306):
                    date_str = exif.get(tag_id)
                    if date_str:
                        return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    
    # Fallback to last modified time if EXIF is missing
    return datetime.fromtimestamp(image_path.stat().st_mtime)

def get_folder_date(folder_path: Path) -> datetime | None:
    """Finds the earliest image date inside a given folder."""
    dates = [
        get_image_date(file)
        for file in folder_path.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTS
    ]
    return min(dates) if dates else None

def sort_and_rename_folders(parent_dir: str, rename: bool = False):
    """Sorts subfolders by their earliest image date and optionally prepends YYYY-MM-DD."""
    base_path = Path(parent_dir)
    if not base_path.is_dir():
        raise ValueError(f"Directory not found: {parent_dir}")

    folder_dates = []
    for item in base_path.iterdir():
        if item.is_dir():
            earliest_date = get_folder_date(item)
            if earliest_date:
                folder_dates.append((earliest_date, item))
            else:
                print(f"Skipping '{item.name}' (no supported images found).")

    # Sort folders chronologically
    folder_dates.sort(key=lambda x: x[0])

    for dt, folder in folder_dates:
        date_prefix = dt.strftime("%Y-%m-%d")
        print(f"{date_prefix} -> {folder.name}")

        if rename and not folder.name.startswith(date_prefix):
            new_name = base_path / f"{date_prefix}_{folder.name}"
            if not new_name.exists():
                folder.rename(new_name)
            else:
                print(f"  Skipped rename: '{new_name.name}' already exists.")

if __name__ == "__main__":
    TARGET_DIRECTORY = "/path/to/your/collection"
    
    # Set rename=False for a dry-run preview; change to True to rename folders on disk
    sort_and_rename_folders(TARGET_DIRECTORY, rename=False)
