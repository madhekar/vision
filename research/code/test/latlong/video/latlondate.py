
import subprocess
import json
def extract_video_metadata(video_path):
    vmdata = (0.0,0.0,"")
    # command to extract specific tags in CSV format
    command = [
        "exiftool",
        "-j",
        "-n",                  # Numeric (decimal) GPS values
        "-GPSLatitude",        # Latitude tag
        "-GPSLongitude",       # Longitude tag
        "-CreateDate",         # Video creation date
        video_path
    ]
    
    try:
        # Run command and capture output
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        if result.returncode == 0:
            d = json.loads(result.stdout)
            vmdata = (d[0]['GPSLatitude'], d[0]['GPSLongitude'], d[0]['CreateDate'] )
            print(f"Success: Metadata saved to {d[0]['GPSLatitude']}")
        else:
            print(f"Error: {result.stderr}")    
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing ExifTool: {e.stderr}")
    except FileNotFoundError:
        print("Error: ExifTool not found. Ensure it is installed and added to your PATH.")
    return vmdata

print(extract_video_metadata("/home/madhekar/Videos/ffmpeg_frames/video_1/VID_20181205_121309.mp4"))        