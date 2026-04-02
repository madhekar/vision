import subprocess
import re
import os
import shutil

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



corp_detect_and_crop_video("/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/57104886521__AEE42CD9-E79B-4F19-A56A-7EA5D6859217.MOV")