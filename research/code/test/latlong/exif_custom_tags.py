import pyexiv2
import os

# Define the image file
image_file = "example.jpg"

# Create a dummy image file for demonstration if it doesn't exist
if not os.path.exists(image_file):
    from PIL import Image
    img = Image.new('RGB', (60, 30), color = 'red')
    img.save(image_file)

# Open the image metadata
metadata = pyexiv2.ImageMetadata(image_file)
metadata.read()

# Define your custom XMP tag and its value
# XMP tags are typically in the format 'Xmp.Namespace.TagName'
custom_tag_key = 'Xmp.dc.MyCustomTag'  # Using 'dc' (Dublin Core) as an example namespace
custom_tag_value = 'This is my custom value.'

# Add the custom tag
# pyexiv2 allows direct assignment for new tags
metadata[custom_tag_key] = custom_tag_value

# Write the updated metadata back to the image file
metadata.write()

print(f"Custom tag '{custom_tag_key}' with value '{custom_tag_value}' added to {image_file}")

# Verify the tag using exiftool (optional, requires exiftool installed)
# import subprocess
# subprocess.run(["exiftool", image_file])