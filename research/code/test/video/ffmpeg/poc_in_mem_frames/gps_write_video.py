import subprocess

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

# --- Usage Example ---
video_file = "/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/IMG_7219.MOV"
latitude = 37.8667  # San Diego
longitude = 122.2622
altitude = 70.231 

add_gps_to_video(video_file, latitude, longitude, altitude)
