
import os
import pandas as pd
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="Zesha-app")  # Replace "my-app" with a descriptive name


def get_lat_long(address):
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return 0.0, 0.0
    

def get_lat_long_series(address):
    location = geolocator.geocode(address)
    if location:
        return pd.Series((location.latitude, location.longitude))
    else:
        return pd.Series((0.0, 0.0))    

def load_location_metadata():
    adf = pd.read_csv("metadata_gen_latlon.csv")
    return adf


df = load_location_metadata()
df.columns = ['name', 'address', 'desc']


address = "1600 Amphitheatre Parkway, Mountain View, CA"
#lat, lon = get_lat_long(address)
#df['lat'], df['lon'] = zip(df.map(get_lat_long(address)))

#df[['lat','lon']] = df.apply(lambda row : get_lat_long('address'), axis=1, result_type='expand')
#df['lat'], df['lon'] = zip(*df['address'].map(get_lat_long), axis=1, result_type="expand")
#df[["lat"]], df[["lon"]] = df[["address"]].apply(get_lat_long_series, axis=1, result_type="expand")

df["lat"], df["lon"] = zip(*df["address"].map(get_lat_long)) 
#df['address'].apply(get_lat_long).to_list()

print(df)
# if coordinates:
#     latitude, longitude = coordinates
#     print(f"Latitude: {latitude}, Longitude: {longitude}")
# else:
#     print("Could not find coordinates for the given address.")