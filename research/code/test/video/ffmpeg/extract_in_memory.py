import subprocess
import shlex
import json
import numpy as np

def get_video_dims(video_path):
    """Uses ffprobe to get the video frame height and width."""
    cmd = f"ffprobe -v quiet -print_format json -select_streams v:0 -show_entries stream=width,height {shlex.quote(video_path)}"
    ffprobe_output = subprocess.check_output(cmd, shell=True).decode('utf-8')
    ffprobe_output = json.loads(ffprobe_output)
    height = ffprobe_output['streams'][0]['height']
    width = ffprobe_output['streams'][0]['width']
    return height, width

def extract_frames_to_numpy(video_path, num_frames=10):
    """Extracts a specified number of frames from a video into a NumPy array."""
    height, width = get_video_dims(video_path)
    
    # Calculate the total number of frames in the video to determine the interval for 10 frames
    # This is a simple approximation; for more accurate frame selection, use frame filters
    cmd_duration = f"ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(video_path)}"
    duration = float(subprocess.check_output(cmd_duration, shell=True).decode('utf-8').strip())

    cmd_fps = f"ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 {shlex.quote(video_path)}"
    fps_str = subprocess.check_output(cmd_fps, shell=True).decode('utf-8').strip().split('/')
    fps = float(fps_str[0]) / float(fps_str[1]) if len(fps_str) == 2 else float(fps_str[0])
    
    total_frames = int(duration * fps)
    frame_interval = max(1, total_frames // num_frames)

    print(f"Video dimensions: {width}x{height}, FPS: {fps:.2f}, Total frames: {total_frames}, Extraction interval: every {frame_interval} frames")

    # FFmpeg command to output raw video frames to stdout
    # The output format is raw video in RGB24 pixel format
    command = [
        'ffmpeg', '-i', video_path,
        '-vf', f'select=not(mod(n\\,{frame_interval}))', # Select frames at interval
        '-frames:v', str(num_frames), # Limit the total number of frames to 10
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        'pipe:'
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    
    frames_array = np.empty((0, height, width, 3), dtype=np.uint8)
    frame_size = height * width * 3 # 3 bytes per pixel for RGB24

    for _ in range(num_frames):
        # Read raw bytes from stdout pipe
        raw_frame = process.stdout.read(frame_size)
        if not raw_frame:
            break
        # Convert bytes to a numpy array
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        frames_array = np.append(frames_array, [frame], axis=0)

    process.stdout.close()
    process.stderr.close()
    process.wait()

    return frames_array

# Example usage:
# video_file = "input_video.mp4"
# frames = extract_frames_to_numpy(video_file, num_frames=10)
# print(f"Shape of extracted frames numpy array: {frames.shape}") 
