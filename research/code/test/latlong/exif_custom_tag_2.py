import subprocess

def add_custom_xmp_tag(image_path, tag_name, tag_value):
    """
    Adds a custom XMP tag to an image using ExifTool.

    Args:
        image_path (str): The path to the image file.
        tag_name (str): The name of the custom XMP tag (e.g., "MyCustomTag").
        tag_value (str): The value to assign to the custom tag.
    """
    # Construct the ExifTool command
    # -overwrite_original_in_place ensures the original file is modified directly
    # -XMP-dc:YourTagNamespace:TagName=Value is the format for XMP custom tags
    command = [
        "exiftool",
        "-overwrite_original_in_place",
        f"-XMP-dc:{tag_name}={tag_value}",
        image_path
    ]

    try:
        # Execute the ExifTool command
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Successfully added custom tag '{tag_name}' with value '{tag_value}' to '{image_path}'.")
        print("ExifTool Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error adding custom tag: {e}")
        print("ExifTool Error Output:")
        print(e.stderr)
    except FileNotFoundError:
        print("Error: ExifTool not found. Please ensure it is installed and in your system's PATH.")

# Example usage:
if __name__ == "__main__":
    image_file = "/Users/emadhekar/Pictures/E5B16BC6-0DF1-4D08-9F4C-77F74349C1C6.jpeg"  # Replace with your image path
    custom_tag_name = "zeshImgType"
    custom_tag_value = "document"

    add_custom_xmp_tag(image_file, custom_tag_name, custom_tag_value)

    # You can verify the tag by running:
    # exiftool -XMP-dc:MyProjectID "path/to/your/image.jpg"
