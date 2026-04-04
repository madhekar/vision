import subprocess

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

# Run the function on the current directory
extract_video_metadata(".")
