import os
import csv


'''
import os

folder_path = './path_to_images'
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

images = [
    f
    for f in os.listdir(folder_path)
    if f.lower().endswith(image_extensions)
]

print(images)

'''
target_folder = r"/Users/bhal/Downloads/parallel_loops_in_python" # Use r'...' for windows paths
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