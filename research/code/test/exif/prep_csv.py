import os
import csv


def create_locations_exif_template(target_folder):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.py')
    for root, dirs, files in os.walk(target_folder):
        csv_path = os.path.join(root, "coordinates.csv")
        
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["SourceFile","GPSLatitude","GPSLongitude"])
                
            # Write files (excluding the csv we are actively generating)
            for file in files:
                if file != "coorditates.csv" and file.lower().endswith(image_extensions):
                    writer.writerow([file, "-", "-"])

create_locations_exif_template("/mnt/zmdata/home-media-app/data/input-data/img/madhekar")
