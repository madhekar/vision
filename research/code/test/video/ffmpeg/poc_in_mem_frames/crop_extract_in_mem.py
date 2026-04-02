import subprocess
import re
import numpy as np

def extract_cropped_frames(video_path, num_frames=10):
    # 1. Automate cropdetect
    detect_cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', 'cropdetect', '-f', 'null', '-'
    ]
    
    # Run, capturing stderr which contains cropdetect output
    result = subprocess.run(detect_cmd, stderr=subprocess.PIPE, text=True)
    
    # 2. Parse the output for the last (most accurate) crop command
    crop_matches = re.findall(r'crop=(\d+:\d+:\d+:\d+)', result.stderr)
    print(f"---> {crop_matches}")
    if not crop_matches:
        raise ValueError("Could not detect crop parameters")
    
    best_crop = crop_matches[-1] # Use the last detected crop
    print(f"-->Detected crop: {best_crop}")

    # 3. Extract 10 frames in memory
    # We use fps filter to spread 10 frames throughout the video
    # extract_cmd = [
    #     'ffmpeg', '-i', video_path,
    #     '-vf', f'crop={best_crop}, select=not(mod(n\\{num_frames}/(duration)))', #fps={num_frames}/(duration)',
    #     '-vsync', 'vfr',
    #     '-vframes', str(num_frames),
    #     '-f', 'rawvideo',
    #     '-pix_fmt', 'rgb24',
    #     'pipe:1'
    # ]

    extract_cmd = [
    'ffmpeg', '-i', video_path,
    '-vf', f'crop={best_crop}, select=not(mod(t,10)),setpts=N/FRAME_RATE/TB', # Fixed filter "select='not(mod(t,10))',setpts=N/FRAME_RATE/TB"
    '-vsync', 'vfr', # Updated from -vsync
    '-vframes', str(num_frames),
    '-f', 'rawvideo',
    '-pix_fmt', 'rgb24',
    'pipe:1'
     ]
    
    # Get width/height from crop string
    w, h, x, y = map(int, best_crop.split(':'))
    
    process = subprocess.Popen(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    print(f"-->{stderr}")
    # 4. Convert raw video pipe to numpy arrays
    video_np = np.frombuffer(stdout, dtype=np.uint8)
    # Shape: (frames, height, width, channels)
    video_np = video_np.reshape((num_frames, h, w, 3))
    
    return video_np

# Example Usage
frames = extract_cropped_frames('/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/IMG_7220.MOV',10)
print(frames.shape) # Output: (10, H, W, 3)
