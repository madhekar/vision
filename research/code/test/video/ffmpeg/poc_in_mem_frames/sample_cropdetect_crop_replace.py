import subprocess
import re
import os
import json
import shutil

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
        if metadata:
            print(f"gps data: {metadata}")
            for entry in metadata:
                lat = entry.get("GPSLatitude")
                lon = entry.get("GPSLongitude")
                alt = entry.get("GPSAltitude")
                return (lat, lon, alt)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None

def add_gps_to_video(video_path, lat, lon, alt=0):
    """
    Writes GPS coordinates to a video file using ExifTool via subprocess.
    """
    # ExifTool expects positive/negative for N/S and E/W
    lat_tag = f"-GPSLatitude={abs(lat)}"
    lat_ref = "-GPSLatitudeRef={}".format('N' if lat >= 0 else 'S')
    lon_tag = f"-GPSLongitude={abs(lon)}"
    lon_ref = "-GPSLongitudeRef={}".format('E' if lon >= 0 else 'W')
    alt_tag = f"-GPSAltitude={alt}"
    
    cmd = [
        "exiftool",
        "-GPSAltitudeRef=0", # 0 = Above Sea Level
        lat_tag,
        lat_ref,
        lon_tag,
        lon_ref,
        alt_tag,
        "-overwrite_original", # Optional: removes _original backup file
        video_path
    ]
    
    try:
        # Run command and wait for it to finish
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully updated {video_path}, result: {result}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")

def corp_detect_and_crop_video(i_vid):
    t_vid = "tvid.mp4"
    try:
        # Step 1: Detect crop parameters
        detect_cmd = [
            "ffmpeg", "-i", i_vid, 
            "-t", "20",  # Analyze first 20 seconds
            "-vf", "cropdetect", 
            "-f", "null", "-"
        ]
        # Capture stderr because FFmpeg prints logs there

        result = subprocess.run(detect_cmd, stderr=subprocess.PIPE, text=True)

        # Extract the last recommended crop value using regex
        # Looks for strings like "crop=1920:800:0:140"
        matches = re.findall(r"crop=\d+:\d+:\d+:\d+", result.stderr)
        if not matches:
            raise ValueError("no crop area found for: {i_vid}.")

        crop_params = matches[-1] # Use the most recent/stable detected value

        # Step 2: Apply the crop
        crop_cmd = [
            "ffmpeg", "-i", i_vid,
            "-vf", crop_params,            
            "-c:a", "copy",  # Copy audio without re-encoding            
            "-map_metadata", "0", #preserve container metadata
            t_vid
        ]
        subprocess.run(crop_cmd, check=True)

        # Step 3: Replace the original file
        #os.replace(t_vid, i_vid)
        if os.path.exists(i_vid):
            os.remove(i_vid)     # Remove file
        
        shutil.move(t_vid, i_vid)

    except Exception as e:
         print(f"error: in croping video {i_vid} error: {e}")


video_file = "/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/57653982982__006BD7FC-6B6C-4D3B-9545-D537D47E9DE6.MOV"
# "/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/IMG_7220.MOV"

lat, lon, alt = get_video_coordinates(video_file)

corp_detect_and_crop_video(video_file)

add_gps_to_video(video_file, lat, lon, alt)