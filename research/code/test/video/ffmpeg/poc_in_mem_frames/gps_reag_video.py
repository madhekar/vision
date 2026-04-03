import subprocess
import json

def get_video_coordinates(video_path):
    # -ee: Extract embedded metadata (required for video GPS tracks)
    # -n:  Output coordinates as decimal degrees (easier for mapping)
    # -j:  Format output as JSON
    # -gps*: Only extract tags starting with "gps"
    command = ["exiftool", "-ee", "-n", "-j", "-gps*", video_path]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        
        # 'metadata' will be a list of dicts. If the video has a GPS track, 
        # it will contain multiple entries for different time steps.
        return metadata
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None

# Usage
video_file = "sample_video.mp4"
gps_data = get_video_coordinates(video_file)

if gps_data:
    for entry in gps_data:
        lat = entry.get("GPSLatitude")
        lon = entry.get("GPSLongitude")
        if lat and lon:
            print(f"Position: {lat}, {lon}")
