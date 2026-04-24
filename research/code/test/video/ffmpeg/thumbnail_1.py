import subprocess

def generate_thumbnail_subprocess(in_path, out_path, time='00:00:05'):
    command = [
        'ffmpeg',
        '-ss', time,
        '-i', in_path,
        '-vframes', '1',
        '-q:v', '2',  # High quality (1-31, lower is better)
        out_path,
        '-y'           # Overwrite output if it exists
    ]
    subprocess.run(command, check=True)

generate_thumbnail_subprocess('input.mp4', 'thumbnail.png')