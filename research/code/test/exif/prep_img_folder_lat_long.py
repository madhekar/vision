import os
import pandas as pd

folder_path = '/path/to/folder'

# Get all file names
files = os.listdir(folder_path)

# Create DataFrame   SourceFile, GPSLatitude, and GPSLongitude
df = pd.DataFrame(files, columns=['SourceFile'])