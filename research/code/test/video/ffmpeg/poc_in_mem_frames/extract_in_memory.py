import subprocess
import shlex
import json
import time
import re
from io import BytesIO
import numpy as np
import base64
from PIL import Image
import face_predictor as fp_tor
import ollama_llava_video_next as olvn

'''
ffmpeg -i "/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/57674026128__645EE475-C9B3-4065-B98B-B8DEBADF0166.MOV" 
-vf cropdetect -f null -

ffmpeg -i /mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/57674026128__645EE475-C9B3-4065-B98B-B8DEBADF0166.MOV 
-vf "crop=720:960:0:0" -c:a copy out.mov
'''

def encode_frames(frame):

    return base64.b64decode(frame).decode("utf-8").strip()

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
        'pipe:'
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    
    bytes_array = []
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
        #im.show()
        #time.sleep(3)
        frames_array = np.append(frames_array, [frame], axis=0)

        buf = BytesIO()
        im.save(buf, format="PNG")
        bytes_array.append(buf.getvalue()) 

    process.stdout.close()
    process.stderr.close()
    process.wait()
    
    print(frames_array.shape)
    return frames_array, bytes_array

# Example usage:
video_file = "/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/IMG_9040.mov"
#"/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/IMG_7811.mov"
#"/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/IMG_7222.mov"
#"/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/IMG_0121.MOV"
# "out.mov"
#"/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/IMG_8137.mp4.mov"
#"/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/IMG_7220.MOV"
#"/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/IMG_7326.mov"
#"/mnt/zmdata/home-media-app/data/input-data/video/madhekar/f12a2136-eec9-5957-8cc8-eb55c6884463/IMG_2245.mov"
#"/mnt/zmdata/home-media-app/data/input-data/video/madhekar/f12a2136-eec9-5957-8cc8-eb55c6884463/IMG_3172.mov"
#"/mnt/zmdata/home-media-app/data/input-data/video/madhekar/f12a2136-eec9-5957-8cc8-eb55c6884463/IMG_3172.mov"
#"/mnt/zmdata/home-media-app/data/input-data/video/madhekar/f12a2136-eec9-5957-8cc8-eb55c6884463/IMG_7284.MOV"
#"/mnt/zmdata/home-media-app/data/input-data/video/madhekar/f12a2136-eec9-5957-8cc8-eb55c6884463/IMG_8016.MOV"
#"/mnt/zmdata/home-media-app/data/input-data/video/madhekar/f12a2136-eec9-5957-8cc8-eb55c6884463/IMG_7717.MOV"
#"/mnt/zmdata/home-media-app/data/input-data/video/madhekar/f12a2136-eec9-5957-8cc8-eb55c6884463/IMG_2069.mov"
#"/home/madhekar/Videos/ffmpeg_frames/video_1/VID_20181205_121309.mp4"

frames, img_bytes_array = extract_frames_to_numpy(video_file, num_frames=10)
print(f"Shape of extracted frames numpy array: {frames.shape}") 

app, svm_classifier, le = fp_tor.init_predictor_module()

detected_persons = []
emotions = []
#frames_enc = [ encode_frames(f) for f in frames]
for nf in range(10):

    img = frames[nf, :, :, :]
    
#     frames_enc.append(encode_frames(img))
#     rgb_img = Image.fromarray(img, "RGB")

#     rgb_img.show()

    llm_partial_pmt = fp_tor.predict_img_faces(app, img, svm_classifier, le)
    p, e = llm_partial_pmt
    if p:
      detected_persons.append(p)
      emotions.append(e)
      #word.strip(' "\'\t\r\n')  re.sub(r"[\s'\"]"," ",word).strip()  word.replace(" ", "").replace("'","").replace('"',"")
for i, v in enumerate(zip(detected_persons, emotions)):
    print(f"frame# {i} person: {v[0]}  emotion: {v[1]}")
ppt = " ".join(dict.fromkeys([ word.strip().strip('"\'').strip().replace('"',"").strip() for word in detected_persons])) + " with emotions " + " ".join(list(set(emotions)))
print(ppt)

print(f"people detected:{detected_persons} with emotion: {emotions}" )

txt = olvn.describe_multiple_images(img_bytes_array, ppt=ppt, location="madhekar residance in san diego, california")

print(txt)
    


    #time.sleep(3)