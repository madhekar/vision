import subprocess
import shlex
import os
from utils.util import model_util as mu
from utils.config_util import config
from utils.util import statusmsg_util as sm
"""
missing-metadata:
  input_image_path: '/home/madhekar/work/home-media-app/data/input-data/img'
  missing_metadata_path: /home/madhekar/work/home-media-app/data/input-data/error/img/missing-data/
  missing_metadata_file: missing-metadata-wip.csv

"""
def execute(source_name):
    sm.add_messages("metadata", "s| starting to analyze missing metadata files...")

    imp, mmp, mmf = config.missing_metadata_config_load()

    input_image_path = os.path.join(imp, source_name)

    args = shlex.split(
        #f"exiftool -GPSLongitude -GPSLatitude -DateTimeOriginal -csv -T -r -n {input_image_path}"
        f"find {input_image_path} -name '*' -print0 | xargs -0 exiftool -GPSLongitude -GPSLatitude -DateTimeOriginal -csv -T -r -n"
    )
    proc = subprocess.run(args, capture_output=True)

    arc_folder_name_dt = mu.get_foldername_by_datetime()
    output_file_path = os.path.join(mmp, source_name, arc_folder_name_dt)
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    with open(os.path.join(output_file_path, mmf), "wb") as output:
        output.write(proc.stdout)

    sm.add_messages("metadata",f"w| finized to analyze missing metadata files created {output_file_path}.",)    
if __name__=='__main__':
    execute(source_name="")