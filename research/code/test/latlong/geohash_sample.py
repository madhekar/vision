
import geohash

lat =51.502278
lon = -0.141947

geohash_cd = geohash.encode(lat, lon, precision=12)

print(f'geohash code for lat =51.502278, lon = -0.141947 {geohash_cd}')

print(f' decode: {geohash.decode(geohash_cd)}')