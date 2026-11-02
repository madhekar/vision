import os
import subprocess
import json

def get_video_format(file_path):
    """Detects the container format of a video using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error', '-show_format', 
        '-show_streams', '-of', 'json', file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Error reading file metadata: {result.stderr}")
        
    metadata = json.loads(result.stdout)
    # Get the format name (e.g., 'mov,mp4,m4a,3gp,3g2,mj2' or 'matroska,webm')
    format_name = metadata.get('format', {}).get('format_name', '')
    return format_name

def convert_to_mp4_if_needed(input_path):
    """Checks if a video is MP4. If not, converts it."""
    if not os.path.exists(input_path):
        print("File does not exist.")
        return

    try:
        file_format = get_video_format(input_path)
        print(f"Detected format: {file_format}")

        # Check if 'mp4' is part of the detected format string
        if 'mp4' in file_format.split(','):
            print(f"'{input_path}' is already an MP4. No conversion needed.")
            return

        # If it's not MP4, prepare output path
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_converted.mp4"
        
        print(f"Converting '{input_path}' to MP4...")
        # '-c:v libx264 -c:a aac' ensures high-compatibility MP4 encoding
        conversion_cmd = [
            'ffmpeg', '-i', input_path, 
            '-c:v', 'libx264', '-c:a', 'aac', 
            '-movflags', '+faststart', output_path
        ]
        
        subprocess.run(conversion_cmd, check=True)
        print(f"Successfully converted! Saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example Usage:
# convert_to_mp4_if_needed("my_video.mkv")
