import ffmpeg
import io
import tempfile
import numpy as np
from PIL import Image
from sentence_transformers import models, SentenceTransformer

video_path = "/home/madhekar/work/zsource/family/video/"
video_name = "IMG_0005.MOV"

target_image_width = 224

img_model = SentenceTransformer(modules=[models.CLIPModel()])


def scaled_size(width, height):
    width_percent = target_image_width / float(width)
    height_size = int((float(height) * float(width_percent)))
    return target_image_width, height_size


def get_frames(content):
    with tempfile.NamedTemporaryFile(mode="w+b") as f:
        f.write(io.BytesIO(content.encode('utf-8')).getbuffer())

        probe = ffmpeg.probe(f.name, threads=1)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        width, height = scaled_size(int(video_info["width"]), int(video_info["height"]))

    out, err = (
        ffmpeg
        .input(f.name, threads=1)
        .filter("scale", width, height)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, quiet=True)
    )
    frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

    indexes = np.random.randint(frames.shape[0], size=10)
    return [io.to_byte_array(frame) for frame in frames[indexes, :]]

if __name__=='__main__':
    print(get_frames(video_path + video_name))


# ffmpeg -i "/home/madhekar/work/zsource/family/video/IMG_0005.MOV" -vf "select=eq(n\,34)" -vframes 1 out.png
# ffmpeg -ss 4 -i /home/madhekar/work/zsource/family/video/IMG_0005.MOV -s 320x240 -frames:v 1 output.jpg
# ffmpeg -i /home/madhekar/work/zsource/family/video/IMG_0005.MOV -r 0.25 output_%04d.png
# ffmpeg -skip_frame nokey -i /home/madhekar/work/zsource/family/video/IMG_0005.MOV -vsync vfr thumb%04d.png -hide_banner
# ffmpeg -skip_frame nokey -i /home/madhekar/work/zsource/family/video/IMG_0005.MOV -vf "select='gt(scene,0.4)'" -vsync vfr -q:v 1 -r 0.05 -s 224x224 frame%04d.jpg

"""
    -vsync vfr: discard the unused frames
    -frame_pts true: use the frame index for image names, otherwise, the index starts from 1 and increments 1 each time
"""