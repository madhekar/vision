import os
from exiftool import ExifToolHelper

# Specify the directory containing your image files
image_directory = "./images"

class ImageBatchProcessor:
    """
    A class to process image files in a directory using ExifTool.
    """
    def __init__(self, directory):
        self.directory = directory
        if not os.path.isdir(self.directory):
            raise FileNotFoundError(f"Directory not found: {self.directory}")

    def get_image_files(self):
        """
        Retrieves a list of supported image files in the directory.
        """
        supported_extensions = ['.jpg', '.jpeg', '.png', '.tif']
        files = []
        for filename in os.listdir(self.directory):
            # Check if the file has a supported image extension
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                files.append(os.path.join(self.directory, filename))
        return files

    def read_metadata_batch(self):
        """
        Reads metadata for all image files in a single batch process.
        """
        image_files = self.get_image_files()
        if not image_files:
            print(f"No image files found in '{self.directory}'.")
            return

        print(f"Processing {len(image_files)} files...")
        
        with ExifToolHelper() as et:
            # Use get_tags to retrieve metadata for all files at once
            all_metadata = et.get_tags(image_files, tags=["FileName", "DateTimeOriginal", "Make", "Model"])
        
        for metadata in all_metadata:
            print("-" * 30)
            print(f"File: {metadata.get('SourceFile', 'N/A')}")
            print(f"  - Original Date: {metadata.get('EXIF:DateTimeOriginal', 'N/A')}")
            print(f"  - Camera Make:   {metadata.get('EXIF:Make', 'N/A')}")
            print(f"  - Camera Model:  {metadata.get('EXIF:Model', 'N/A')}")

# --- Example Usage ---
if __name__ == "__main__":
    processor = ImageBatchProcessor(image_directory)
    processor.read_metadata_batch()