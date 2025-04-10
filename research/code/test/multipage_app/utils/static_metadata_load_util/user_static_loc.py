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


def random_user_agent(num_chars=8):
    return "".join(random.choices(string.ascii_letters + string.digits, k=8))

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

def join_state_county(line):
    arr1 = line.strip().split(",")
    arr = [e.strip() for e in arr1]
    if len(arr) <= 2:
        return ",".join(arr).strip()
    else:
        l1 = ",".join(arr[-2:]).strip()
        l2 = " ".join(arr[0:2])
        return l2 + "," + l1

def collect_addresses(df, loc_accuracy=0):
    df = df[['latitude', 'longitude']].astype(float).round(loc_accuracy)
    # df["latitude"] = df["latitude"].astype(float).round(2)
    # df["longitude"] = df["longitude"].astype(float).round(4)
    # df['latitude'] = [round(float(x), 4) for x in df['latitude']]
    # df['longitude'] = [round(float(x), 4) for x in df['longitude']]
    df_nodup = df.drop_duplicates(subset=["latitude", "longitude"], keep=False)
    print("--->", df_nodup.size, df_nodup.head())

    df.to_csv("lat_lon_nodup.csv", index=False, encoding="utf-8")

    list_lat_lon = df_nodup.values.tolist()
    result_list = []
    for ll in list_lat_lon:
        time.sleep(5)
        print((ll[1], ll[0]))
        addr = getLocationDetails((ll[0], ll[1]), 3)
        addr = join_state_county(addr)
        print(f"{addr},{ll[1]},{ll[0]}")
        address, state, contry = addr.split(',')
        result_list.append([address, state, contry, ll[1], ll[0]])
    df = pd.DataFrame(result_list, columns=['address','state','county','latitude','longitude'])
    return df


def join_last_four(line):
    arr1 = line.strip().split(",")
    arr = [e.strip() for e in arr1]
    if len(arr) <= 4:
        return ",".join(arr).strip()
    else:
        l1 = ",".join(arr[-4:]).strip()
        l2 = " ".join(arr[0:4])
        return l2 + "," + l1
    
def format_comma_locations(draft_file):
    nlines = []
    #in_file = "data/lat_lon_nodup_full.csv"
    with open(draft_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            nline = join_last_four(line)
            print(nline)
            nlines.append(nline + "\n")

    #out_file = "data/name_lat_lon_full.csv"
    with open(draft_file, "w") as fo:
        fo.writelines(nlines)

def get_unique_locations(df_mis, df_def):

    # remove missing lat lon rows
    condition = df_mis["GPSLatitude"] == "-"
    dfp = df_mis[~(condition)]

    print("not missing-->", dfp.size)

    # match column names to parquet file standards
    dfp.rename(
        columns={"GPSLatitude": "latitude", "GPSLongitude": "longitude"}, inplace=True
    )
    print(dfp.head(), dfp.size)

    # extract non prepresented data from static metadata 
    df_notin2 = dfp[~(dfp['latitude'].isin(df_def['latitude']))]
    print('not in default location files -->', df_notin2.size, df_notin2.head())

    # drop any duplicate data
    dfr= df_notin2.drop_duplicates(subset=['latitude', 'longitude'])

    columns_to_drop = ['SourceFile', 'DateTimeOriginal']
    dfr.drop(columns=columns_to_drop, inplace=True)

    print('after duplicate lat lon drop-->',dfr.size, dfr.head())

    # get location address/ name from library, round lat lon to 6 digits max
    df_ret = collect_addresses(dfr)

    return df_ret

def load_missing_recods(f_name, static_metadata):

   df = pd.read_csv(f_name)
   print('missing metadata -->', df.size, df.head())

   # remove missing lat lon rows
   condition = df['GPSLatitude'] == '-'
   dfp = df[~(condition)]

   print('not missing-->', dfp.size)

   # match column names to parquet file standards
   dfp.rename(columns={'GPSLatitude':'latitude', 'GPSLongitude': 'longitude'}, inplace=True)

   print(dfp.head(),dfp.size)
   
   # read parquet file already generated from default static locations
   dfpq = pd.read_parquet(static_metadata)

   print('parquet -->', dfpq.size, dfpq.head())

   # extract non prepresented data from static metadata 
   df_notin2 = dfp[~(dfp['latitude'].isin(dfpq['latitude']))]
   print('not in parquet file -->', df_notin2.size, df_notin2.head())

   # drop any duplicate data
   dfr= df_notin2.drop_duplicates(subset=['latitude', 'longitude'])

   columns_to_drop = ['SourceFile', 'DateTimeOriginal']
   dfr.drop(columns=columns_to_drop, inplace=True)

   print('after duplicate lat lon drop-->',dfr.size, dfr.head())

   # get location address/ name from library, round lat lon to 6 digits max
   collect_addresses(dfr)

   dfr.to_csv('new_lat_lon.csv', index=False, encoding='utf-8')

# static_metadata = '/home/madhekar/work/home-media-app/data/app-data/static-metadata/static_locations.parquet'
# input_file = "/home/madhekar/work/home-media-app/data/input-data-1/error/img/missing-data/AnjaliBackup/missing-metadata-wip.csv"
# load_missing_recods(input_file, static_metadata)


## 2
# nlines = []
# in_file = "data/lat_lon_nodup_full.csv"
# with open(in_file, "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         nline = join_last_four(line)
#         print(nline)
#         nlines.append(nline + "\n")

# out_file = "data/name_lat_lon_full.csv"
# with open(out_file, "w") as fo:
#     fo.writelines(nlines)

## 1
# file = "new_lat_lon.csv"
# df = pd.read_csv(file)
# collect_addresses(df)


