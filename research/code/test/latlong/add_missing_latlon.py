import os
import pandas as pd
import pyexiv2
from PIL import Image
import streamlit as st
from GPSPhoto import gpsphoto
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import datetime
import random
import string
import time

cache = {}
# get location address information from latitude and longitude
def getLocationDetails(strLnL, max_retires):
    address = "n/a"

    if strLnL in cache:
        return cache[strLnL]
    else:
        geolocator = Nominatim(user_agent=random_user_agent())
        try:
            rev = RateLimiter(geolocator.reverse, min_delay_seconds=1)
            location = rev(strLnL, language="en", exactly_one=True)
            if location:
                address = location.address
                cache[strLnL] = address
                return address
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            st.warning(f"Get address failed with {e}")
    return address


def random_user_agent(num_chars=8):
    return "".join(random.choices(string.ascii_letters + string.digits, k=8))
   
def collect_addresses(df):
    df['latitude'] = df['latitude'].round(6)
    df["longitude"] = df["longitude"].round(6)
    df_nodup = df.drop_duplicates(subset=['latitude', 'longitude'], keep=False)
    print('--->', df_nodup.size, df_nodup.head())

    df.to_csv('lat_lon_nodup.csv', index=False, encoding='utf-8')

    list_lat_lon = df_nodup.values.tolist()
    result_list = []
    for ll in list_lat_lon:
        time.sleep(5)
        addr = getLocationDetails((ll[1],ll[0]), 3)
        print(f'{addr},{ll[1]},{ll[0]}')
        result_list.append([addr, ll[1], ll[0]])
    df = pd.DataFrame(result_list, columns=['address', 'latitude', 'longitude'])    
    
    df.to_csv("new_address_lat_long.csv", index=False, encoding="utf-8")



# def load_missing_recods(f_name, static_metadata):
   
#    df = pd.read_csv(f_name)
#    print('missing metadata -->', df.size, df.head())

#    condition = df['GPSLatitude'] == '-'
#    dfp = df[~(condition)]

#    print('not missing-->', dfp.size)

#    dfp.rename(columns={'GPSLatitude':'latitude', 'GPSLongitude': 'longitude'}, inplace=True)

#    print(dfp.head(),dfp.size)

#    dfpq = pd.read_parquet(static_metadata)

#    print('parquet -->', dfpq.size, dfpq.head())

#    df_notin2 = dfp[~(dfp['latitude'].isin(dfpq['latitude']))]
#    print('not in parquet file -->', df_notin2.size, df_notin2.head())

#    dfr= df_notin2.drop_duplicates(subset=['latitude', 'longitude'])

#    columns_to_drop = ['SourceFile', 'DateTimeOriginal']

#    dfr.drop(columns=columns_to_drop, inplace=True)

#    print('after duplicate lat lon drop-->',dfr.size, dfr.head())

#    collect_addresses(dfr)
#    dfr.to_csv('new_lat_lon.csv', index=False, encoding='utf-8')

# static_metadata = '/home/madhekar/work/home-media-app/data/app-data/static-metadata/static_locations.parquet'
# input_file = "/home/madhekar/work/home-media-app/data/input-data-1/error/img/missing-data/AnjaliBackup/missing-metadata-wip.csv"   
# load_missing_recods(input_file, static_metadata)

file = 'new_lat_lon.csv'
df = pd.read_csv(file)
collect_addresses(df)
