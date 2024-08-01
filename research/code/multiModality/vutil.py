import ffmpeg
import io
import numpy as np
from PIL import Image
from sentence_transformers import models, SentenceTransformer

target_image_width = 224

img_model = SentenceTransformer(modules=[models.CLIPModel()])

def scaled_size(width, height):
  width_percent = (target_image_width / float(width))
  height_size = int((float(height) * float(width_percent)))
  return target_image_width, height_size

def get_frames(content):
  with tempfile.NamedTemporaryFile(delete_on_close=True) as f:
    f.write(io.BytesIO(content).getbuffer())

    probe = ffmpeg.probe(f.name, threads=1)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width, height = scaled_size(int(video_info['width']), int(video_info['height']))

  out, err = (
    ffmpeg
    .input(f.name, threads=1)
    .filter('scale', width, height)
    .output('pipe', format='rawvideo', pix_fmt='rgb24')
    .run(capture_stdout=True, quiet=True)
  )
  frames = (
    np
    .frombuffer(out, np.uint8)
    .reshape([-1, height, width, 3])
  )

  indexes = np.random.randint(frames.shape[0], size=10)
  return [ io.to_byte_array(frame) for frame in frames[indexes, :] ]

def get_embeddings(frames):
    images = [Image.open(io.BytesIO(frame)) for frame in frames]
    vectors = img_model.encode(images)
    return [vector.tolist() for vector in vectors]