import whisper
import ffmpeg
import os

def generate_subtitles(video_path, output_srt_path, model_size="base"):
    """
    Generates an SRT subtitle file from a video's audio track using OpenAI Whisper.
    """
    print(f"Loading Whisper model: {model_size}...")
    model = whisper.load_model(model_size)
    print("Model loaded. Transcribing audio...")

    # The transcribe method handles audio extraction internally
    result = model.transcribe(video_path, fp16=False) # fp16=False for CPU compatibility

    # Write the transcription to an SRT file
    with open(output_srt_path, "w", encoding="utf-8") as f:
        # Whisper's result object has a built-in method to create SRT
        f.write(result["srt"]) # Note: result["srt"] is a custom addition by some whisper utility wrappers. 
        # A more standard way might involve iterating through segments and formatting manually.

    print(f"Subtitles saved to {output_srt_path}")

def add_subtitles_to_video(video_path, subtitle_path, output_video_path):
    """
    Adds (burns in or soft subs) the generated SRT file to the video using FFmpeg.
    This example burns them in (hard subtitles) for universal playback.
    """
    print("Adding subtitles to the video...")
    try:
        # Use a video filter 'vf' to apply the subtitles
        ffmpeg.input(video_path).output(
            output_video_path,
            vf=f"subtitles={subtitle_path}",
            acodec='copy' # copy audio codec to avoid re-encoding audio
        ).run(overwrite_output=True)
        print(f"Subtitled video saved to {output_video_path}")
    except ffmpeg.Error as e:
        print('FFmpeg Error:', e.stderr.decode('utf8'))

# Example usage:
if __name__ == "__main__":
    input_video = "my_video.mp4" # Replace with your video file path
    output_srt = "output_subtitles.srt"
    output_video = "video_with_subtitles.mp4"
    
    # Check if input file exists
    if not os.path.exists(input_video):
        print(f"Error: {input_video} not found.")
    else:
        generate_subtitles(input_video, output_srt)
        add_subtitles_to_video(input_video, output_srt, output_video)
