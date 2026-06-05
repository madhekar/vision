import subprocess
import shutil

def scale_video_to_1080p(input_path: str, output_path: str):
    # Verify that ffmpeg is installed and available in the system PATH
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg executable not found. Please install it.")

    # Assemble the FFmpeg command components into a list
    command = [
        "ffmpeg",
        "-y",                             # Overwrite the output file if it already exists
        "-i", input_path,                 # Specify the path to the input video file
        "-vf", "scale=-2:360",          # Scale filter: maintains aspect ratio, sets height to 1080p
        "-c:v", "libx264",                # Encode the video stream using the H.264 codec
        "-preset", "slow",                # Speed/compression tradeoff preset (e.g., medium, fast, ultrafast)
        "-crf", "22",                     # Constant Rate Factor controlling quality (18-28 is standard)
        "-c:a", "copy",                   # Directly copy the audio track without re-encoding to save time
        output_path                       # Specify the path for the scaled output video file
    ]

    try:
        # Run the command and capture standard error if the process fails
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Success! Scaled video saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during video scaling. FFmpeg exited with code {e.returncode}.")
        print(f"FFmpeg Error Output:\n{e.stderr}")

# Example usage
if __name__ == "__main__":
    scale_video_to_1080p("/home/madhekar/tmp/VID_20181205_171018.mp4", "/home/madhekar/tmp/output_1080p_video.mp4")
