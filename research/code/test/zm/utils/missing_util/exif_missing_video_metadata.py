import subprocess
import os
from utils.config_util import config as cfg
from utils.util import storage_stat as ss
from utils.missing_util import exif_missing_metadata as emm
from utils.util import statusmsg_util as sm

def extract_video_metadata(directory_path, output_csv="video_metadata.csv"):
    # command to extract specific tags in CSV format
    command = [
        "exiftool",
        "-csv",                # Output in CSV format
        "-n",                  # Numeric (decimal) GPS values
        "-GPSLatitude",        # Latitude tag
        "-GPSLongitude",       # Longitude tag
        "-CreateDate",         # Video creation date
        "-ext", "mp4",         # Filter for .mp4 files
        "-ext", "mov",         # Filter for .mov files
        "-ext", "avi",         # Filter for .avi files
        directory_path
    ]

    try:
        # Run command and capture output
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Write the captured output directly to your file
        with open(output_csv, "w") as f:
            f.write(result.stdout)
            
        print(f"Success: Metadata saved to {output_csv}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing ExifTool: {e.stderr}")
    except FileNotFoundError:
        print("Error: ExifTool not found. Ensure it is installed and added to your PATH.")

def get_missing_metadata_dataframe(fname):
    df = pd.read_csv(fname)
    dfr = df[
        (
            (df["SourceFile"] != "SourceFile")
            & (df["GPSLongitude"] != "GPSLongitude")
            & (df["GPSLatitude"] != "GPSLatitude")
            & (df["DateTimeOriginal"] != "DateTimeOriginal")
        )
    ]
    dfr.to_csv(fname)
    return dfr.values.tolist()

def execute(source_name):

    sm.add_messages("metadata", "s| starting to analyze missing metadata files...")

    imp, vmp, mmp, mvmp, mmf, mvmf, mmff, mvmff = cfg.missing_metadata_config_load()

    input_video_path = os.path.join(vmp, source_name)
    #clean empty folders if any
    #ss.remove_empty_files_and_folders(input_image_path) #remove_empty_folders(input_image_path) 

    output_file_path = os.path.join(mvmp, source_name)
    ss.create_folder(output_file_path)
        
    out_file = os.path.join(output_file_path, mvmf)    
    
    extract_video_metadata(input_video_path, out_file)

    df = get_missing_metadata_dataframe(out_file)

    filter_missing_video_data(os.path.join(output_file_path, mvmf), os.path.join(output_file_path, mvmff))

    create_missing_report(os.path.join(output_file_path, mvmf))


