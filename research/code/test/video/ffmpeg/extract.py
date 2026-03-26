import ffmpeg
import os

def extract_frames_with_fixed_width(video_path, output_dir, num_frames, target_width):
    """
    Extracts a fixed number of frames from a video, scaled to a specific width, 
    using ffmpeg-python.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
        num_frames (int): The desired fixed number of frames to extract.
        target_width (int): The desired width for the output frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Use ffmpeg.probe to get video information (like duration, width, etc.)
        probe = ffmpeg.probe(video_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        if video_stream is None:
            print("No video stream found.")
            return

        # Calculate time intervals for evenly spaced frames (optional, but good for distribution)
        duration = float(video_stream['duration'])
        interval_duration = duration / num_frames
        
        # Prepare and run the ffmpeg command
        # The 'scale' filter adjusts dimensions while preserving aspect ratio with '-1'
        # '-frames:v' limits the output to the exact number of frames specified
        # '-r 1' ensures one frame is output per second of *selected* content (used with setpts)

        # For simple fixed number of evenly spaced frames (using fps filter)
        output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
        
        (
            ffmpeg
            .input(video_path)
            .filter('scale', target_width, -1) # Scale to target_width, auto height
            .output(output_pattern,
                    r=1,          # Output rate of 1 frame per second
                    frames=num_frames, # Limit total frames output
                    f='image2')   # Output format as images
            .run(overwrite_output=True)
        )
        print(f"Successfully extracted {num_frames} frames to {output_dir}")

    except ffmpeg.Error as e:
        print('FFmpeg Error:', e.stderr.decode('utf8'))
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# Replace 'input_video.mp4' with your video file path
# Replace 'output_frames' with your desired output directory name
# Set the number of frames and target width
extract_frames_with_fixed_width('input_video.mp4', 'output_frames', num_frames=10, target_width=640)
