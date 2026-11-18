# from geopy.geocoders import Nominatim

# geolocator = Nominatim(user_agent="my_postal_lookup_app")
# location = geolocator.geocode({"postalcode": "92121", "country": "US"})

# if location:
#     print(location.latitude, location.longitude, location.address)


import pgeocode

nomi = pgeocode.Nominatim('in')
result = nomi.query_postal_code("411030")

print(result.state_code)  # Output: 'CA'
print(result.state_name)  # Output: 'California'
print(result.country_code)