from geopy.geocoders import Nominatim
import sys

def getll(address):
   loc = Nominatim(user_agent='lon')
   getLoc = loc.geocode(address)
   return getLoc.latitude,getLoc.longitude

if __name__=='__main__':
   if len(sys.argv) > 1:
      address = sys.argv[1:]
      print(getll(address))