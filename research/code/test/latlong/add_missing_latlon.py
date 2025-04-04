import os
import pandas as pd

def load_missing_recods(f_name, static_metadata):
   
   df = pd.read_csv(f_name)
   print('missing metadata -->', df.size, df.head())

   condition = df['GPSLatitude'] == '-'
   dfp = df[~(condition)]

   
   print('not missing-->', dfp.size)

   dfp.rename(columns={'GPSLatitude':'latitude', 'GPSLongitude': 'longitude'}, inplace=True)

   print(dfp.head(),dfp.size)

   dfpq = pd.read_parquet(static_metadata)

   print('parquet -->', dfpq.size, dfpq.head())

   df_notin2 = dfp[~(dfp['latitude'].isin(dfpq['latitude']))]
   print('not in parquet file -->', df_notin2.size, df_notin2.head())

   dfr= df_notin2.drop_duplicates(subset=['latitude', 'longitude'])

   columns_to_drop = ['SourceFile', 'DateTimeOriginal']

   dfr.drop(columns=columns_to_drop, inplace=True)

   print('after duplicate lat lon drop-->',dfr.size, dfr.head())

   dfr['name'] = '???'
   dfr['state'] = '??'
   dfr['country'] = '??'
   dfr.to_csv('new_lat_lon.csv', index=False, encoding='utf-8')

static_metadata = '/home/madhekar/work/home-media-app/data/app-data/static-metadata/static_locations.parquet'
input_file = "/home/madhekar/work/home-media-app/data/input-data-1/error/img/missing-data/AnjaliBackup/missing-metadata-wip.csv"   
load_missing_recods(input_file, static_metadata)
