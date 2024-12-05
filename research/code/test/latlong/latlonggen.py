

from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="my-app")  # Replace "my-app" with a descriptive name


def get_lat_long(address):
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None


address = "1600 Amphitheatre Parkway, Mountain View, CA"
coordinates = get_lat_long(address)

if coordinates:
    latitude, longitude = coordinates
    print(f"Latitude: {latitude}, Longitude: {longitude}")
else:
    print("Could not find coordinates for the given address.")