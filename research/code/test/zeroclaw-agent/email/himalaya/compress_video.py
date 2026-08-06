import sys
import os
import subprocess
import json

def get_duration(input_file):
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'json', input_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    data = json.loads(result.stdout)
    return float(data['format']['duration'])

def compress_video(input_file, target_size_mb):
    # Setup filenames
    name, ext = os.path.splitext(input_file)
    output_file = f"{name}_compressed.mp4"
    
    # Calculate target bitrate
    duration = get_duration(input_file)
    audio_bitrate_kbps = 128
    
    # Formula: (Target Size in bits / duration) - audio bitrate
    target_size_bits = target_size_mb * 1024 * 1024 * 8
    total_bitrate_bps = target_size_bits / duration
    video_bitrate_bps = total_bitrate_bps - (audio_bitrate_kbps * 1000)
    
    video_bitrate_kbps = int(video_bitrate_bps / 1000)
    
    if video_bitrate_kbps <= 0:
        print("Error: Target size is too small for this video duration.")
        return

    print(f"Targeting: {target_size_mb} MB | Video Bitrate: {video_bitrate_kbps} kbps")

    # Pass 1
    pass1_cmd = [
        'ffmpeg', '-y', '-i', input_file, 
        '-c:v', 'libx264', '-b:v', f'{video_bitrate_kbps}k',
        '-pass', '1', '-an', '-f', 'mp4', os.devnull if os.name != 'nt' else 'NUL'
    ]
    
    # Pass 2
    pass2_cmd = [
        'ffmpeg', '-y', '-i', input_file, 
        '-c:v', 'libx264', '-b:v', f'{video_bitrate_kbps}k',
        '-pass', '2', '-c:a', 'aac', '-b:a', f'{audio_bitrate_kbps}k', 
        output_file
    ]

    print("Running Pass 1...")
    subprocess.run(pass1_cmd)
    
    print("Running Pass 2...")
    subprocess.run(pass2_cmd)
    
    # Clean up two-pass log files
    for file in os.listdir('.'):
        if file.startswith('ffmpeg2pass'):
            os.remove(file)
            
    print(f"Finished! Saved as: {output_file}")

    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compress.py <input_video> <target_size_in_mb>")
    else:
        compress_video(sys.argv[1], float(sys.argv[2]))