import pandas as pd
import os
import subprocess
# Path to the CSV file and the photo directory
'''
    Photograph,Y,X
    photo1.jpg,37.7749,-122.4194
    photo2.jpg,34.0522,-118.2437
'''
csv_path = r"..\location.csv"
photos_dir = r"..\photofolder\photos"
exiftool_path = r"..\location_of_exiftool\exiftool.exe"

# Read the CSV file, assuming the delimiter is a comma
data = pd.read_csv(csv_path, delimiter=',')
# Remove any leading or trailing spaces from column names
data.columns = data.columns.str.strip()
# Print column names to verify they are read correctly
print("Column names:", data.columns)
# Check the first few rows to verify the data
print(data.head())
for index, row in data.iterrows():
    latitude = row['Y']
    longitude = row['X']
    photo = row['Photograph']
    
    # Construct the full path to the photo
    photo_path = os.path.join(photos_dir, photo)
    
    # ExifTool command
    # The latitude and longitude are in WGS84 coordinates
    command = [
        exiftool_path,
        f"-GPSLatitude={latitude}",
        f"-GPSLatitudeRef={'N' if latitude >= 0 else 'S'}",
        f"-GPSLongitude={longitude}",
        f"-GPSLongitudeRef={'E' if longitude >= 0 else 'W'}",
        "-overwrite_original",  # Overwrite the original file to avoid duplication
        photo_path
    ]
    
    # Run the command
    subprocess.run(command)
print("Batch geotagging completed.")