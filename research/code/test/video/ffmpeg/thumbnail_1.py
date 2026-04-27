import subprocess

'''
The thumbnail filter in FFmpeg is an algorithmic tool designed to select the most "representative" frame from a specified sequence of frames. Unlike a simple frame extraction (like taking the first frame), it analyzes a batch of frames and picks the one that best captures the visual essence of that segment. 
How It Works
The filter analyzes a cluster of n consecutive frames and calculates an average histogram for the entire group. It then selects the single frame whose individual histogram has the smallest "sum of squared errors" compared to that average. This process effectively avoids picking outlier frames, such as sudden flashes, solid black frames, or blurry transitions, in favor of a frame that is most typical of the scene. 
Basic Usage
The primary parameter is n, which defines the number of frames to analyze in each batch. 

    Extract one representative frame every 100 frames:
    bash

    ffmpeg -i input.mp4 -vf "thumbnail=100" -frames:v 1 output.png

    This command looks at the first 100 frames, picks the "best" one, and saves it as a single image.
    Continuous extraction across a video:
    bash

    ffmpeg -i input.mp4 -vf "thumbnail=n=100" -vsync 0 out%d.png

    This will output one image for every 100 frames of the video. 

Key Features and Considerations

    Representative Selection: It is often used to generate "meaningful" thumbnails by avoiding black or low-detail frames.
    Performance: Because it must analyze every frame in the batch (calculating histograms for each), it is computationally slower than simple seeking or the select filter.
    Chaining Filters: You can combine it with other filters, such as scale, to resize the representative frame immediately.
    bash

    ffmpeg -i input.mp4 -vf "thumbnail=100,scale=320:-1" -vframes 1 thumb.jpg

    Limitations: It can be "tricky" for generating large storyboards as it increases processing time significantly compared to taking every Nth frame. 

For more advanced needs, such as adding the resulting image back into the video file as an album art cover, you can use the -disposition:v attached_pic command. For comprehensive details, refer to the FFmpeg Filters Documentation. 

'''

'''
o scale down a video in FFmpeg on Ubuntu, use the -vf scale filter. A common approach to resize to a specific width (e.g., 720px) while maintaining the original aspect ratio is: ffmpeg -i input.mp4 -vf "scale=720:-1" -c:a copy output.mp4. Use -2 instead of -1 if the encoder requires dimensions to be divisible by 2. 
Common Scaling Commands

    Scale to 720p width (maintain aspect ratio):
    bash

    ffmpeg -i input.mp4 -vf "scale=1280:-2" output.mp4

    Scale to 480p height (maintain aspect ratio):
    bash

    ffmpeg -i input.mp4 -vf "scale=-2:480" output.mp4

    Scale to exact dimensions (may distort image):
    bash

    ffmpeg -i input.mp4 -vf "scale=640:480" output.mp4

     

Best Practices for Quality & Size

    Maintain Aspect Ratio: Use -1 or -2 in the scale filter to automatically calculate the height or width.
    Optimize File Size: Add -c:v libx264 -crf 28 for better compression (higher CRF means lower quality/smaller size).
    Ensure Compatibility: The -2 option ensures the auto-calculated dimension is divisible by two, which is required by many codecs (like H.264/AAC). 

Specific Scenarios

    Downscale 4K to 1080p:
    bash

    ffmpeg -i input_4k.mp4 -vf "scale=1920:1080" output_1080p.mp4

    Scale and Limit Maximum Size (e.g., to 1080p, without upscaling smaller videos):
    bash

    ffmpeg -i input.mp4 -vf "scale='min(1920,iw)':-2" output.mp4

     

This video demonstrates how to downscale a 4K video to 1080p using FFmpeg:
Related video thumbnail
42s
Downscaling 4K / UHD Video To 1080P (FHD) Using FFMPEG ...
Daniel's Tech World
YouTube• Feb 27, 2023
Note: The -c:a copy option can be used to copy the audio stream without re-encoding it, which saves time. 

'''


def generate_scale_down_video(in_path, out_path):

    """ffmpeg -i input.mp4 -vf "scale=-2:480" output.mp4"""
    command = [
        'ffmpeg',
        '-i', in_path,
        '-vf', "scale=-2:200",
        '-q:v', '2',  # High quality (1-31, lower is better)
        out_path,
        '-y'           # Overwrite output if it exists
    ]
    subprocess.run(command, check=True)


def generate_thumbnail_subprocess(in_path, out_path, time='00:00:05'):
    command = [
        'ffmpeg',
        '-ss', time,
        '-i', in_path,
        '-vf', "thumbnail=100,scale=200:-1",
        '-vframes', '1',
        '-q:v', '2',  # High quality (1-31, lower is better)
        out_path,
        '-y'           # Overwrite output if it exists
    ]
    subprocess.run(command, check=True)

generate_thumbnail_subprocess('/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/IMG_9040.mov', 'thumbnail_1.png')

#generate_scale_down_video('/mnt/zmdata/home-media-app/data/input-data/video/Berkeley/794131d8-f8b3-5535-8f14-b9712e2c5169/IMG_9040.mov', 'scale_down.mov')