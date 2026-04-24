import ffmpeg

def create_thumbnail(input_file, output_file, timestamp='00:00:01.000'):
    try:
        (
            ffmpeg
            .input(input_file, ss=timestamp)
            .filter('scale', 200, -1)  # Scale to width 320, keep aspect ratio
            .output(output_file, vframes=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Thumbnail saved to {output_file}")
    except ffmpeg.Error as e:
        print(e.stderr.decode())

create_thumbnail('/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/IMG_9040.mov', 'thumb.jpg')