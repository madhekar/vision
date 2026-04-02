import ffmpeg
import numpy as np

def crop_and_extract_frames(input_path, num_frames=10):
    # 1. Automate crop detection (analyzes first few seconds)
    probe = ffmpeg.probe(input_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    
    # Run cropdetect to find optimal crop parameters
    crop_info = (
        ffmpeg
        .input(input_path)
        .filter('cropdetect', 24, 2)
        .output('null', f='null')
        .run_async(pipe_stderr=True)
    )
    
    # Parse the last cropdetect log line for parameters
    _, stderr = crop_info.communicate()
    last_line = stderr.decode().splitlines()[-1]
    crop_params = last_line.split('crop=')[1].split(' ')[0] # e.g., 1920:1080:0:0

    # 2. Extract 10 frames from the cropped video in memory
    # Create expression to select 10 frames evenly spaced
    vframes_expr = f'if(eq(mod(n,round(n/10)),0),1,0)' 
    
    out, _ = (
        ffmpeg
        .input(input_path)
        .filter('crop', *crop_params.split(':'))
        .filter('select', f'eq(mod(n,ceil(n/{num_frames})),0)') # Simple selection
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=num_frames)
        .run(capture_stdout=True)
    )
    
    # 3. Convert to NumPy Array
    width = int(crop_params.split(':')[0])
    height = int(crop_params.split(':')[1])
    frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    
    return frames

# Usage
# frames = crop_and_extract_frames('input.mp4')
# print(frames.shape) # (10, height, width, 3)
