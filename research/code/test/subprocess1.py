from pathlib import Path
import subprocess, shlex

# Get lines from text file
with open('paths.txt', 'r') as file:
    lines = file.readlines()
    print(lines)
# Get a list of file paths from the dir paths
fp_list = [fp for l in lines if not l.startswith('#') for fp in Path(l.strip()).iterdir()]
print(fp_list)
# The pathlib equivalent to os.listdir() is iterdir()
args = shlex.split('exiftool -p $directory;$filename;$GPSLongitude#;$GPSLatitude#;$GPSAltitude#;$DateTimeOriginal;$iso;$ExposureTime;$Fnumber -f -c "%f"')
# Pass all of the file paths in one ExifTool command
proc = subprocess.run([args, *fp_list], capture_output=True)
# run() replaced call(), run() returns the CompletedProcess class
# Write the headers and ExifTool output to a csv
with open("output.csv", "wb") as output:
    output.write("id;name;E;N;Alt_gps;date;iso;speed;oberture\n".encode())
    # Setting capture_output=True captures output in stdout and stderr
    output.write(proc.stdout)
