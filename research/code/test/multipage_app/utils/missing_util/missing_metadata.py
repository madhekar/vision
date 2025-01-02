import subprocess
import shlex
import os
from utils.config_util import config
from utils.util import statusmsg_util as sm
"""
missing-metadata:
  input_image_path: '/home/madhekar/work/home-media-app/data/input-data/img'
  missing_metadata_path: /home/madhekar/work/home-media-app/data/input-data/error/img/missing-data/
  missing_metadata_file: missing-metadata-wip.csv

"""
def execute():
    sm.add_messages("metadata", "s| starting to analyze missing metadata files...")

    imp, mmp, mmf = config.missing_metadata_config_load()

    args = shlex.split(
        f"exiftool -GPSLongitude -GPSLatitude -DateTimeOriginal -csv -T -r -n {imp}"
    )
    proc = subprocess.run(args, capture_output=True)

    output_file_path = os.path.join(mmp, mmf)
    with open(output_file_path, "wb") as output:
        output.write(proc.stdout)

    sm.add_messages("metadata",f"w| finized to analyze missing metadata files created {output_file_path}.",)    
if __name__=='__main__':
    execute()