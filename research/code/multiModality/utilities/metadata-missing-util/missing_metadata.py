import subprocess
import shlex
import os
import yaml
import pprint
"""
missing-metadata:
  input_image_path: '/home/madhekar/work/home-media-app/data/input-data/img'
  missing_metadata_path: /home/madhekar/work/home-media-app/data/input-data/error/img/missing-data/
  missing_metadata_file: missing-metadata-wip.csv

"""
def extract_missing_metadata(root_folder, output_path, output_file_name):
   
   args = shlex.split(f"exiftool -GPSLongitude -GPSLatitude -DateTimeOriginal -csv -T -r -n {root_folder}")
   proc = subprocess.run(args, capture_output=True)
   
   output_file_path = os.path.join(output_path, output_file_name)
   with open(output_file_path, "wb") as output:
      output.write(proc.stdout)

def config_load():
    with open("missing_metadata_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        pprint.pprint("* * * * * * * * * * * Missing Metadata Properties * * * * * * * * * * * *")
        pprint.pprint(dict)
        pprint.pprint("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
        input_image_path = dict["missing-metadata"]["input_image_path"]
        missing_metadata_path = dict["missing-metadata"]["missing_metadata_path"]
        missing_metadata_file = dict["missing-metadata"]["missing_metadata_file"]
    return (
        input_image_path,
        missing_metadata_path,
        missing_metadata_file,
    )

if __name__=='__main__':
    imp, mmp, mmf = config_load()

    extract_missing_metadata(imp, mmp, mmf)