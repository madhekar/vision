import os
from PIL import Image
from PIL.ExifTags import TAGS

# Your image path
image_file = "/Users/emadhekar/tmp/images/IMAG2293.jpg"

# Open the image file
image = Image.open(image_file)

# Make a collection of properties and values corresponding to your image.
exif = {}
if image._getexif() is not None:
    for tag, value in image._getexif().items():
        if tag in TAGS:
            exif[TAGS[tag]] = value

if "GPSInfo" in exif:
    gps_info = exif["GPSInfo"]

    def convert_to_degrees(value):
        """
        Helper function to convert the GPS coordinates stored in the EXIF to degrees in float format.

        Args:
            value (tuple): The GPS coordinate as a tuple (degrees, minutes, seconds)

        Returns:
            float: The coordinate in degrees
        """
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)

    # Convert latitude and longitude to degrees
    lat = convert_to_degrees(gps_info[2])
    lon = convert_to_degrees(gps_info[4])
    lat_ref = gps_info[1]
    lon_ref = gps_info[3]

    # Adjust the sign of the coordinates based on the reference (N/S, E/W)
    if lat_ref != "N":
        lat = -lat
    if lon_ref != "E":
        lon = -lon

    # Format the GPS coordinates into a human-readable string
    geo_coordinate = "{0}° {1}, {2}° {3}".format(lat, lat_ref, lon, lon_ref)
    print(geo_coordinate)

    # Create a Google Maps link
    google_maps_link = f"https://www.google.com/maps?q={lat},{lon}"
    print(f"Google Maps link: {google_maps_link}")
else:
    print("No GPS information found.")