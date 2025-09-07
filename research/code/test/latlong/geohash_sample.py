
from geohash import encode, decode

lat =51.502278
lon = -0.141947

geohash_cd = encode(lat, lon, precision=12)

print(f'geohash code for lat =51.502278, lon = -0.141947 {geohash_cd}')

print(f' decode: {decode(geohash_cd)}')