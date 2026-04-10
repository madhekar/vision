import subprocess
import shlex
import json
import uuid
import os
from io import BytesIO
import numpy as np
import base64
import PIL
from PIL import Image
import asyncio
import multiprocessing as mp
import pandas as pd
import aiomultiprocess as aiomp
from aiomultiprocess import Pool
import aiofiles

from functools import partial
from utils.config_util import config
from utils.util import location_util as lu
from utils.util import model_util as mu
from utils.util import fast_parquet_util as fpu
import face_predictor as fp_tor
import ollama_llava_video_next as olvn
import streamlit as st

'''
ffmpeg -i "/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/57674026128__645EE475-C9B3-4065-B98B-B8DEBADF0166.MOV" 
-vf cropdetect -f null -

ffmpeg -i /mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/57674026128__645EE475-C9B3-4065-B98B-B8DEBADF0166.MOV 
-vf "crop=720:960:0:0" -c:a copy out.mov
'''

PIL.Image.MAX_IMAGE_PIXELS = 933120000
d_latitude, d_longitude = 32.968689, -117.184243
d_loc = 'Madhekar residence in Carmel Valley'

# init LLM modules
@st.cache_resource
def location_initialize(smp, smf):
    try:
        df = fpu.init_location_cache(os.path.join(smp, smf))
    except Exception as e:
        st.error(f"exception occurred in loading location metadata: {smf} with exception: {e}")  
    return df 

@st.cache_resource
def location_initialize_btree(smp, smf):
    try:
        btree_ = fpu.init_location_btree_cache(os.path.join(smp, smf))
    except Exception as e:
        st.error(f"exception occurred in loading location metadata: {smf} with exception: {e}")  
    return btree_ 
    
def get_loc_name_by_latlon(latlon):
    nm = st.session_state.ball_tree.get_location_name_for_latlong(latlon[0], latlon[1])
    print(f'lat-lon: {latlon} => loc name: {nm}')
    return nm

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

def describe_video(args):
    location, uri = args
    frames, img_bytes_array = extract_frames_to_numpy(uri, num_frames=10)
    print(f"Shape of extracted frames numpy array: {frames.shape}") 

    app, svm_classifier, le = fp_tor.init_predictor_module()

    detected_persons = []
    emotions = []
    for nf in range(10):
        img = frames[nf, :, :, :]
        
        llm_partial_pmt = fp_tor.predict_img_faces(app, img, svm_classifier, le)
        p, e = llm_partial_pmt
        if p:
          detected_persons.append(p)
          emotions.append(e)

    frames_people_emo ="following people found in video frames: "      
    for i, v in enumerate(zip(detected_persons, emotions)):
        frames_people_emo += f" image {i + 1} {v[0]} in {v[1]} mood, "

    print(frames_people_emo)  
    txt = olvn.describe_multiple_images(img_bytes_array, ppt=frames_people_emo, location=location)

    print(f"video description: {txt}")

    return txt
    
# get location details as: (latitude, longitude) and address
async def locationDetails(args, lock):
    uri = args
    async with lock:
      try:  
        #print(f"-->> {uri}")
        loc = ""
        lat_lon = ()
        lat_lon = lu.extract_video_latlon(uri)
        print(f' {lat_lon} : {uri}')
        if lat_lon == ():
            lat_lon = (d_latitude, d_longitude) # default location lat, lon
            loc = d_loc # default location description
        else:    
           loc = get_loc_name_by_latlon(lat_lon)
        #print(f"--->> {lat_lon} : {loc}")
        if not loc:
            loc = lu.getLocationDetails(lat_lon, max_retires=3)
      except Exception as e:
        st.error(f'exception: {e} occurred in getting lat/ lon or location details for {uri}')
      return str(lat_lon), loc
        
# uuid4 id for vector database
async def generateId(args):
    uri= args
    return str(uuid.uuid4()) #[uri, str(uuid.uuid4())]

async def timestamp(args):
    uri = args
    print(f'ts--{uri}')
    ts,uc = lu.extract_video_timestamp(uri) #lu.get_image_exif_info(uri) 
    print(f'*** timestamp: {ts} user comment: {uc}')
    return str(ts), str(uc)

async def caption(args):
    uri = args
    cap = await olvn.caption_image(uri)
    return cap 

# appends json rows to file
async def append_file(filename, dict_data_list, mode):
    async with aiofiles.open(filename, mode) as f:
        for dict_element in dict_data_list:
           stv = json.dumps(dict_element)
           #info(stv)
           await f.write(stv)
           await f.write(os.linesep)
        await f.close()       

"""
multi processing linux tools
ps -ef | grep -w streamlit
pgrep --count streamlit
killall -9 streamlit
"""
async def run_workflow(
    df,
    video_dir_path,
    chunk_size,
    queue_size,
    metadata_path,
    metadata_file,
    num_files,
    number_of_instances,
    openclip_finetuned
):
    st.info(f"CPU COUNT: {chunk_size}")
    
    progress_generation = st.sidebar.empty()
    bar = st.sidebar.progress(0)
    if df is not None:
      if not df.empty:
        num = df.shape[0]
      else:
        num=0
    else:
      num = 0
     
    # num_files = len(glob.glob(os.path.join(image_dir_path,'/**/*')))
    num = num_files - num
    print(f'processing files in: {video_dir_path} total files: {num_files}') 
    

    lock = asyncio.Lock()
    img_iterator = mu.getRecursive(video_dir_path, chunk_size=chunk_size)

    with st.status("Generating LLM responses...", expanded=True) as status:
        async with Pool(processes=chunk_size,  queuecount=queue_size, maxtasksperchild=1) as pool:  #initializer=pool_init, initargs=(bfs,),
            count = 0
            res = []
            for ilist in img_iterator:
                rlist = mu.is_processed_batch(ilist, df)
                print(rlist)
                if len(rlist) > 0:
                    res = await asyncio.gather(
                        pool.map(generateId, rlist),
                        pool.map(timestamp, rlist),
                        pool.map(partial(locationDetails, lock=lock), rlist),
                        pool.map(caption, rlist),
                    )
                    
                    res.append(rlist)

                    print('*===>', res)
                    rflist, oflist = new_xform(res)

                    res1 = await asyncio.gather(pool.map(describe_video,  rflist))
                    #res1 = await pool.map(describeImage,  rflist)
                    print('****', res1)

                    zlist = [oflist[i] + [res1[0][i]]  for i in range(len(oflist))]

                    fdictlist = final_xform(zlist)

                    st.info(fdictlist)

                    await append_file(os.path.join(metadata_path, metadata_file), fdictlist, 'a+')

                count = count + len(ilist)
                count = num if count > num else count
                if num > 0:
                    progress_generation.text(f"{count} files processed out-of {num} => {int((100 / num) * count)}% processed")
                    bar.progress(int((100 / num) * count))
                else:
                    progress_generation.text(f"{count} files processed out-of {num} => {int((100 / 1) * count)}% processed (all done!)")
                    bar.progress(int((100 / 1) * count))    
        #st.info(res)
        pool.close()
        # pool.join()

    status.update(label="process completed!", state="complete", expanded=False)

def execute(user_source_selected):

    #if device == "cuda:0":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    #mp.freeze_support()
    aiomp.set_start_method("fork")
    (
        image_dir_path,
        video_dir_path,
        metadata_path,
        metadata_file,
        chunk_size,
        number_of_instances,
        openclip_finetuned,
        static_metadata_path,
        static_metadata_file
    ) = config.preprocess_config_load()

    static_metadata_path = os.path.join(static_metadata_path, user_source_selected)
    # add user data source to image input and metadata output paths
    video_dir_path = os.path.join(video_dir_path, user_source_selected)
    number_of_files = mu.count_files_in_path(video_dir_path)

    if not os.path.exists(video_dir_path):
        st.error(f'exception: image data path for {user_source_selected} does not exists!')
    metadata_path = os.path.join(metadata_path, user_source_selected)
    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)

    if "df_loc" not in st.session_state:
        df = location_initialize(static_metadata_path, static_metadata_file)
        st.session_state.df_loc = df
    else:
        df_loc = st.session_state.df_loc   

    if "ball_tree" not in st.session_state:
        btree = location_initialize_btree(static_metadata_path, static_metadata_file)
        st.session_state.ball_tree = btree
    else:
        ball_tree = st.session_state.ball_tree        

    #btree = location_initialize(static_metadata_path, static_metadata_file)    

    chunk_size = int(mp.cpu_count() // 4)
    queue_size = chunk_size
    st.sidebar.subheader("Metadata Generation")
    st.sidebar.divider()

    df = None
    try:
        if os.path.exists(os.path.join(metadata_path, metadata_file)):
            data = []
            with open(os.path.join(metadata_path, metadata_file), mode="r") as f:
                res = f.read()
                res = res.replace("\n", "")
                res = res.replace("}{", "},{")
                res = "[" + res + "]"
                data = json.loads(res)
                df = pd.DataFrame(data)
                df = mu.drop_except(df, ["uri"])
    except Exception as e:
        st.error(f"exception: {e} occurred in loading metadata file")

    # bcreate_metadata = st.button("start metadata creation")
    # if bcreate_metadata:

    print(df)

    try:
        asyncio.run(run_workflow(
            df,
            video_dir_path,
            chunk_size,
            queue_size,
            metadata_path,
            metadata_file,
            number_of_files,
            number_of_instances,
            openclip_finetuned,
        ))
    except Exception as e:
        st.error(f'exception: {e} occurred in async main function')    


# kick-off metadata generation
if __name__ == "__main__":
    execute("user_source_selected")

    #time.sleep(3)
    location = "Madhekar residance San Diego, california"
    # # Example usage:
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
    describe_video(video_file, location )


