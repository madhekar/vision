
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
         loc = geolocator.geocode(row.address, exactly_one=True, timeout=60)
         try:
           d['lat'] = loc.latitude
         except Exception as e: 
            pass
         try:
            d['lon'] = loc.longitude
         except Exception as e:
            pass

         time.sleep(2)
         locs.append(d)

     pprint(locs)       

def simple_with_error_chk_details():
    df = pd.read_csv("default-addresses.csv")
    locs = []
    i = 1
    for row in df.itertuples():
        d = {}
        try:
         loc = geolocator.geocode(row.address, exactly_one=True, timeout=60, addressdetails=True)
         data = loc.raw
         data = data['address']
         state = country = "" 
         state = str(data['state']).lower()
         country = str(data['country_code']).lower()
         d['id'] = i
         d['desc'] = row.name
         d['state'] = state
         d['country'] = country
         d["lat"] = loc.latitude
         d["lon"] = loc.longitude
         locs.append(d)
         i+=1
        except Exception as e:
           pprint(f'error: geocode failed with {row.adress} with exception {e.message}')
        time.sleep(2)

    dfr = pd.DataFrame(locs, columns=['id', 'country', 'state', 'desc','lat', 'lon']).set_index('id', drop=True)    
    pprint(dfr.head())


simple_with_error_chk_details()