
import pandas as pd


def extract_stats_of_metadata_file(metadata_path):
    mdf = pd.read_csv(metadata_path)
    print(mdf.head(10))
    count_missing_latlong = mdf.groupby(['GPSLatitude'], as_index=False)['-'].sum()
    count_missing_datetime = mdf.groupby(['DateTimeOriginal'], as_index=False)['-'].sum()
    count_missing_latlong_and_datetime = mdf.groupby(["DateTimeOriginal", 'GPSLatitude'], as_index=False)["-"].sum()
    return {
        "latlong": count_missing_latlong,
        "datetime": count_missing_datetime,
        "latlong_dt": count_missing_latlong_and_datetime
    }
    '''
    /home/madhekar/work/home-media-app/data/input-data/error/img/missing-data/missing-metadata-wip.csv
    '''

    if __init__=='__mail__':
       dict =  extract_stats_of_metadata_file('/home/madhekar/work/home-media-app/data/input-data/error/img/missing-data/missing-metadata-wip.csv')

       print(dict)