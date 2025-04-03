import pyexiv2 as pe
"""
>>> metadata.exif_keys [
 'Exif.Image.ImageDescription',
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
 'Exif.Photo.PixelYDimension'
 ]

Exif.Image.GPSTag: <Exif.Image.GPSTag [Long] = 1502>
Exif.GPSInfo.GPSLatitudeRef: <Exif.GPSInfo.GPSLatitudeRef [Ascii] = N>
Exif.GPSInfo.GPSLatitude: <Exif.GPSInfo.GPSLatitude [Rational] = 62/1 19/1 153497/10000>
Exif.GPSInfo.GPSLongitudeRef: <Exif.GPSInfo.GPSLongitudeRef [Ascii] = W>
Exif.GPSInfo.GPSLongitude: <Exif.GPSInfo.GPSLongitude [Rational] = 150/1 6/1 260595/10000>
Exif.GPSInfo.GPSAltitudeRef: <Exif.GPSInfo.GPSAltitudeRef [Byte] = 0>
Exif.GPSInfo.GPSAltitude: <Exif.GPSInfo.GPSAltitude [Rational] = 109800/1000>
Exif.GPSInfo.GPSTimeStamp: <Exif.GPSInfo.GPSTimeStamp [Rational] = 17/1 22/1 10/1>
Exif.GPSInfo.GPSProcessingMethod: <Exif.GPSInfo.GPSProcessingMethod [Comment] = ASCII>
Exif.GPSInfo.GPSDateStamp: <Exif.GPSInfo.GPSDateStamp [Ascii] = 2014:06:23>

"""

# Open an image file
image_path = "/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup/bf98198d-fcc6-51fe-a36a-275c06005669/IMAG0191.jpg"
metadata = pe.ImageMetadata(image_path)

# Read existing EXIF data
metadata.read()

#print(metadata.exif_keys)

# # Print all EXIF tags and their values
print("Original EXIF data:")
# for tag in metadata.exif_keys:
#     print(f"{tag}: {metadata[tag]}")

print(metadata['Exif.Image.ImageDescription'])
print(metadata["Exif.GPSInfo.GPSLatitude"], metadata["Exif.GPSInfo.GPSLatitudeRef"])
print(metadata["Exif.GPSInfo.GPSLongitude"], metadata["Exif.GPSInfo.GPSLongitudeRef"])
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