
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import pandas as pd
from pprint import pprint

d = []
geolocator = Nominatim(user_agent="name_ofgent")

# def load_addresses():
    
#     return df

# def extract_latlon(df):
#    for row in df.itertuples():
#      dt = {}
#      location = geolocator.geocode(row.address, namedetails=True) 
#      pprint(location)
#      dt['lat'] = location.latitude
#      dt['lon'] = location.longitude
#      dt['name'] = row.name
#     #  dt['country'] = location.country
#     #  dt['state'] = location.state
#      d.append(dt)
#    return d 
#    y = df['address'].apply(lambda x:geolocator.geocode(x, namedetails=True))
#    #location = geolocator.geocode(Address, namedetails=True)
#    #print((location.latitude, location.longitude))
#    print(y.latitude)

def df_apply_with_ratelimiting():
    df = pd.read_csv("default-addresses.csv")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=5)
    df['coord'] = df['address'].apply(geocode).apply(lambda location: (location.latitude, location.longitude))
    return df


def simple_with_error_chk():
     df = pd.read_csv("default-addresses.csv")
     locs = []
     for row in df.itertuples():
         d = {}
         pprint(row)
         l = geolocator.geocode(row.address, exactly_one=True, timeout=60)
         try:
           d['lat'] = l.latitude
         except: 
            pass
         try:
            d['lon'] = l.longitude
         except:
            pass

         time.sleep(2)
         locs.append(d)

     pprint(locs)              

simple_with_error_chk()