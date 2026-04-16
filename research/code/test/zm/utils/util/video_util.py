import subprocess
import re
import os
import shutil
import shlex
import json
import time
import numpy as np
from PIL import Image

'''
ffmpeg -i "/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/57674026128__645EE475-C9B3-4065-B98B-B8DEBADF0166.MOV" 
-vf cropdetect -f null -

ffmpeg -i /mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/57674026128__645EE475-C9B3-4065-B98B-B8DEBADF0166.MOV 
-vf "crop=720:960:0:0" -c:a copy out.mov
'''

def get_video_dims(video_path):
    """Uses ffprobe to get the video frame height and width."""
    cmd = f"ffprobe -v quiet -print_format json -select_streams v:0 -show_entries stream=width,height {shlex.quote(video_path)}"
    ffprobe_output = subprocess.check_output(cmd, shell=True).decode('utf-8')
    ffprobe_output = json.loads(ffprobe_output)
    height = ffprobe_output['streams'][0]['height']
    width = ffprobe_output['streams'][0]['width']
    print(f"video dimentions {height}:{width}")
    return height, width

def extract_frames_to_numpy(video_path, num_frames=10):
    """Extracts a specified number of frames from a video into a NumPy array."""
    height, width = get_video_dims(video_path)
    
    # Calculate the total number of frames in the video to determine the interval for 10 frames
    # This is a simple approximation; for more accurate frame selection, use frame filters
    cmd_duration = f"ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(video_path)}"
    duration = float(subprocess.check_output(cmd_duration, shell=True).decode('utf-8').strip())
    print(f"video duration: {duration}")

    cmd_fps = f"ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 {shlex.quote(video_path)}"
    fps_str = subprocess.check_output(cmd_fps, shell=True).decode('utf-8').strip().split('/')
    fps = float(fps_str[0]) / float(fps_str[1]) if len(fps_str) == 2 else float(fps_str[0])
    
    total_frames = int(duration * fps)
    frame_interval = max(1, total_frames // num_frames)
    print(f"video fps: {fps} total frames: {total_frames} frame interval: {frame_interval}")

    print(f"Video dimensions: {width}x{height}, FPS: {fps:.2f}, Total frames: {total_frames}, Extraction interval: every {frame_interval} frames")

    """
    # 3. Use ffmpeg to extract frames
        ffmpeg_cmd = [
            'ffmpeg', '-y', # -y to overwrite output files
            '-i', video_path,
            '-r', str(calculated_fps), # Set the calculated rate
            "-loglevel", "error",
            output_pattern
        ]
    """

    # FFmpeg command to output raw video frames to stdout
    # The output format is raw video in RGB24 pixel format
    command = [
        'ffmpeg', '-i', video_path,
        '-vf', f'select=not(mod(n\\,{frame_interval}))', # Select frames at interval
        '-vsync', 'vfr',
        '-frames:v', str(num_frames), # Limit the total number of frames to 10
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        "-loglevel", "error",
        'pipe:'
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    
    frames_array = np.empty((0, height, width, 3), dtype=np.uint8)
    frame_size = height * width * 3 # 3 bytes per pixel for RGB24
    print(f"video frame size: {frame_size}")

    for _ in range(num_frames):
        # Read raw bytes from stdout pipe
        raw_frame = process.stdout.read(frame_size)
        if not raw_frame:
            break
        # Convert bytes to a numpy array
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        im = Image.fromarray(frame)
        im.show()
        time.sleep(3)
        frames_array = np.append(frames_array, [frame], axis=0)

    process.stdout.close()
    process.stderr.close()
    process.wait()

    return frames_array

# def corp_detect_and_crop_video(i_vid):
#     t_vid = "tvid.mp4"
#     try:
#         # Step 1: Detect crop parameters
#         detect_cmd = [
#             "ffmpeg", "-i", i_vid, 
#             "-t", "20",  # Analyze first 20 seconds
#             "-vf", "cropdetect", 
#             "-f", "null", "-"
#         ]
#         # Capture stderr because FFmpeg prints logs there

#         result = subprocess.run(detect_cmd, stderr=subprocess.PIPE, text=True)

#         # Extract the last recommended crop value using regex
#         # Looks for strings like "crop=1920:800:0:140"
#         matches = re.findall(r"crop=\d+:\d+:\d+:\d+", result.stderr)
#         if not matches:
#             raise ValueError("no crop area found for: {i_vid}.")

#         crop_params = matches[-1] # Use the most recent/stable detected value

#         # Step 2: Apply the crop
#         crop_cmd = [
#             "ffmpeg", "-i", i_vid,
#             "-vf", crop_params,
#             "-c:a", "copy",  # Copy audio without re-encoding
#             t_vid
#         ]
#         subprocess.run(crop_cmd, check=True)

#         # Step 3: Replace the original file
#         #os.replace(t_vid, i_vid)
#         if os.path.exists(i_vid):
#             os.remove(i_vid)     # Remove file
        
#         shutil.move(t_vid, i_vid)

#     except Exception as e:
#          print(f"error: in croping video {i_vid} error: {e}")



####


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
'''
If FFmpeg fails to find a crop area (e.g., black bars) with cropdetect, the video likely lacks sufficient black borders, has uneven black levels, 
or the sampling time is too early. Fix this by testing later in the video, increasing the limit of black pixel density, or manually setting
 dimensions.Key Troubleshooting Steps:Sample Later in the Video: The default check often happens at the start (e.g., logos). Use -ss 60 to 
 check a later part where black bars are visible:ffmpeg -ss 60 -i input.mp4 -vframes 10 -vf cropdetect -f null -.Increase Limit (Lower Sensitivity): 
 If black bars are dirty, increase the threshold (0-255). 
 A higher value makes it less strict about absolute black:ffmpeg -i input.mp4 -vf "cropdetect=limit=30:round=2" -f null -.Manually Set Crop: If auto-detection fails, 
 manually use the crop filter (width:height:x:y):ffmpeg -i input.mp4 -vf "crop=1920:800:0:140" output.mp4.Avoid Invalid Sizes: 
 Ensure width/height are even numbers and not larger than the source video.Common Command Pattern:bash# 1. Detect
ffmpeg -i input.mp4 -vf "cropdetect=24:16:0" -f null -
# 2. Apply (replace w:h:x:y with output from step 1)
ffmpeg -i input.mp4 -vf "crop=w:h:x:y" -c:a copy output.mp4
``` [10]
FFMPEG cropdetect and crop filters the inside scoopMar 8, 2013 — hello everyone I've got a neat example uh in regards to using FFM Peg and specifically
the crop detect. and the crop feature that'11:53YouTube·techtiptricksIs there is any way ffmpeg can remove the black bars around this videoApr 29, 2022 — 
Then throw that in your ffmpeg command when converting it. ... If commands aren't your thing - give Shutter Encoder a try (basical...RedditRemove .mp4 
video top and bottom black bars using ffmpegSep 11, 2014 — To remove black bars from the top and bottom of a video using ffmpeg, you can use the cropdetect 
filter: 1. Use `ffmpeg -ss 90 -i ...Super UserShow all

'''


def corp_detect_and_crop_video(i_vid):
    t_vid = "tvid.mp4"
    try:
        # Step 1: Detect crop parameters
        detect_cmd = [
            "ffmpeg", "-i", i_vid, 
            "-t", "20",  # Analyze first 20 seconds
            "-vf", "cropdetect=limit=64",  #increase detection sinsitivity
            "-loglevel", "error",
            "-f", "null", "-"
        ]
        # Capture stderr because FFmpeg prints logs there

        result = subprocess.run(detect_cmd, stderr=subprocess.PIPE, text=True)
        print(f"video: {i_vid} *** {result.stdout} *** {result.stderr}")

        # Extract the last recommended crop value using regex
        # Looks for strings like "crop=1920:800:0:140"
        matches = re.findall(r"crop=\d+:\d+:\d+:\d+", result.stderr)
        if not matches:
            raise ValueError(f"no crop area found for: {i_vid}. {result.stdout} ::: {result.stderr} ")
        
        crop_params = matches[-1] # Use the most recent/stable detected value
        
        
        # add fps before crop
        crop_params = "fps=30," + crop_params
        print(f"+++>crops parameters: {crop_params} found for: {i_vid}")

        # Step 2: Apply the crop
        crop_cmd = [
            "ffmpeg", "-i", i_vid,
            "-vf", crop_params,            
            "-c:a", "copy",  # Copy audio without re-encoding            
            "-map_metadata", "0", #preserve container metadata
            "-loglevel", "error",
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


def crop_detect_and_crop_video_workaround(i_vid):
  try:
    #lat, lon, alt = get_video_coordinates(i_vid)

    corp_detect_and_crop_video(i_vid)

    #add_gps_to_video(i_vid, lat, lon, alt)  

  except Exception as e:
      print(f"Error in crop_detect_and_crop_video_workaround {i_vid} : {e}")


