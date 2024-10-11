import subprocess, shlex, os


def extract_missing_metadata(rood_folder, output_path, output_file_name):
   
   args = shlex.split("exiftool -GPSLongitude -GPSLatitude -DateTimeOriginal -csv -T -r -n '%root_folder'")
   proc = subprocess.run(args, capture_output=True)
   
   output_file_path = os.path.join(output_path, output_file_name)
   with open(output_file_path, "wb") as output:
      output.write(proc.stdout)

