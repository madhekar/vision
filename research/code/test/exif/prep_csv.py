import os
import csv

target_folder = r"/Users/bhal/Downloads/parallel_loops_in_python" # Use r'...' for windows paths

for root, dirs, files in os.walk(target_folder):
    csv_path = os.path.join(root, "coordinates.csv")
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["SourceFile","GPSLatitude","GPSLongitude"])
            
        # Write files (excluding the csv we are actively generating)
        for file in files:
            if file != "coorditates.csv":
                writer.writerow([file, "-", "-"])