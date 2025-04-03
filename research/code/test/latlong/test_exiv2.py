import pyexiv2

# Open an image file
image_path = "image.jpg"
metadata = pyexiv2.ImageMetadata(image_path)

# Read existing EXIF data
metadata.read()

# Print all EXIF tags and their values
print("Original EXIF data:")
for tag, entry in metadata.exif_items:
    print(f"{tag}: {entry.value}")

# Modify a specific EXIF tag
key = 'Exif.Image.Make'
metadata[key] = 'NewCameraMake'

# Write the modified metadata back to the image
metadata.write()

# Re-read the metadata to verify changes
metadata = pyexiv2.ImageMetadata(image_path)
metadata.read()

# Print the modified EXIF data
print("\nModified EXIF data:")
for tag, entry in metadata.exif_items:
    print(f"{tag}: {entry.value}")