import subprocess
import math

def extract_fixed_frames(video_path, total_desired_frames=10, output_pattern="frame_%04d.jpg"):
    # 1. Get video duration using ffprobe
    probe_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    duration = float(subprocess.check_output(probe_cmd).decode().strip())
    
    # 2. Calculate frames per second (fps) needed
    # Rate = total frames / duration
    calculated_fps = total_desired_frames / duration
    
    # 3. Use ffmpeg to extract frames
    ffmpeg_cmd = [
        'ffmpeg', '-y', # -y to overwrite output files
        '-i', video_path,
        '-r', str(calculated_fps), # Set the calculated rate
        output_pattern
    ]
    
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Extracted approximately {total_desired_frames} frames.")

# Example Usage
extract_fixed_frames("/mnt/zmdata/home-media-app/data/input-data/video/madhekar/f12a2136-eec9-5957-8cc8-eb55c6884463/IMG_2069.mov", 10) 
