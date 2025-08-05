import subprocess
import shlex
import os
import pandas as pd
from utils.util import model_util as mu
from utils.config_util import config
from utils.util import statusmsg_util as sm
from utils.util import storage_stat as ss
"""
missing-metadata:
  input_image_path: '/home/madhekar/work/home-media-app/data/input-data/img'
  missing_metadata_path: /home/madhekar/work/home-media-app/data/input-data/error/img/missing-data/
  missing_metadata_file: missing-metadata-wip.csv
  missing_metadata_filter_file: missing-metadata-filter-wip.csv
  find /home/madhekar/work/home-media-app/data/input-data/img -name '*' -print0 | xargs -0 exiftool -gps:GPSLongitude -gps:GPSLatitude -DateTimeOriginal -csv -T -r -n
  f"find '{input_image_path}' -name '*' -print0 | xargs -0 exiftool -GPSLongitude -GPSLatitude -DateTimeOriginal -csv -T -r -n"
"""

def create_missing_report(missing_file_path):
    df = pd.read_csv(missing_file_path)
    n_total = len(df)
    #df = format_lat_lon(df)
    if df.index.size > 0:
        n_lon = len(df[df["GPSLongitude"] == "-"])
        n_lat = len(df[df["GPSLatitude"] == "-"])
        n_dt = len(df[df["DateTimeOriginal"] == "-"])

        sm.add_messages( "metadata", f"w| missing data Longitudes: {n_lon} Latitude: {n_lat} DataTime: {n_dt} of: {n_total} rows")
    else:
        ss.remove_file(missing_file_path)
        sm.add_messages(
            "metadata",
            "w| missing data Longitudes: 0 Latitude: 0 DataTime: 0 of: 0 rows",
        )    

def filter_missing_image_data(missing_file_path, missing_filter_file_path):
    df = pd.read_csv(missing_file_path)
    print('----->', df.index.size)
    #df = format_lat_lon(df)
    if (df.index.size) > 0:
       dfm = df[(df['GPSLatitude'] == '-') | (df['GPSLongitude'] == '-') | (df['DateTimeOriginal'] == '-')]  
       dfm.to_csv(missing_filter_file_path,sep=',')
    else:
        sm.add_messages('metadata', f'e| empty or invalid missing metadata file {missing_file_path}') 
          

def execute(source_name):
    sm.add_messages("metadata", "s| starting to analyze missing metadata files...")

    imp, mmp, mmf, mmff= config.missing_metadata_config_load()

    input_image_path = os.path.join(imp, source_name)
    #clean empty folders if any
    #ss.remove_empty_files_and_folders(input_image_path) #remove_empty_folders(input_image_path) 
    
    try:            
        args = shlex.split( f"exiftool -gps:GPSLongitude -gps:GPSLatitude -DateTimeOriginal -csv -T -r -n {input_image_path}")
        proc = subprocess.run(args, capture_output=True)
    except Exception as e:
        print(f'error {e}')

    # print(proc.stderr)
    print(proc.stdout)

    #arc_folder_name_dt = mu.get_foldername_by_datetime()

    output_file_path = os.path.join(mmp, source_name) #, arc_folder_name_dt)

    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    
    with open(os.path.join(output_file_path, mmf), "wb") as output:
        output.write(proc.stdout)

    filter_missing_image_data(os.path.join(output_file_path, mmf), os.path.join(output_file_path, mmff))

    create_missing_report(os.path.join(output_file_path, mmf))

    sm.add_messages("metadata",f"w| finized to analyze missing metadata files created {output_file_path}.",)    
if __name__=='__main__':
    execute(source_name="")