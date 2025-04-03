import pyexiv2 as pe
'''
>>> metadata.exif_keys ['Exif.Image.ImageDescription',
 'Exif.Image.XResolution',
 'Exif.Image.YResolution',
 'Exif.Image.ResolutionUnit',
 'Exif.Image.Software',
 'Exif.Image.DateTime',
 'Exif.Image.Artist',
 'Exif.Image.Copyright',
 'Exif.Image.ExifTag',
 'Exif.Photo.Flash',
 'Exif.Photo.PixelXDimension',
 'Exif.Photo.PixelYDimension']
'''

# Open an image file
image_path = "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/bf98198d-fcc6-51fe-a36a-275c06005669/IMAG0191.jpg"
metadata = pe.ImageMetadata(image_path)

# Read existing EXIF data
metadata.read()

print(metadata.exif_keys)

# # Print all EXIF tags and their values
print("Original EXIF data:")
for tag in metadata.exif_keys:
    print(f"{tag}: {metadata[tag]}")

# # Modify a specific EXIF tag
# key = 'Exif.Image.ImageDescription'
# metadata[key] = 'San diego school 6th grade'

# # Write the modified metadata back to the image
# metadata.write()

# # Re-read the metadata to verify changes
# metadata = pe.ImageMetadata(image_path)
# metadata.read()

# # Print the modified EXIF data
# print("\nModified EXIF data:")
# for tag, entry in metadata.exif_keys:
#     print(f"{tag}: {entry.value}")