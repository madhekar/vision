import sys
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS, IFD
from GPSPhoto import gpsphoto
from geopy.geocoders import Nominatim
    
def extract(img):
 
  # open the image
  image = Image.open(img)
 
  # extracting the exif metadata
  exifdata = image.getexif()
 
  # looping through all the tags present in exifdata
  for tagid in exifdata:
    #print(tagid, ", ", TAGS.get(tagid, tagid))
    # getting the tag name instead of tag id
    tagname = TAGS.get(tagid, tagid)
 
    # passing the tagid to get its respective value
    value = exifdata.get(tagid)
   
    # printing the final result
    print(f"{tagname:25}: {value}")

  for ifd_id in IFD:
        print('>>>>>>>>>', ifd_id.name, '<<<<<<<<<<')
        try:
            ifd = exif.get_ifd(ifd_id)

            if ifd_id == IFD.GPSInfo:
                resolve = GPSTAGS
            else:
                resolve = TAGS

            for k, v in ifd.items():
                tag = resolve.get(k, k)
                print(tag, v)
        except KeyError:
            pass  

def getDateTime(img):
    # open the image
    image = Image.open(img)

    # extracting the exif metadata
    exifdata = image.getexif()
    date_time = exifdata.get(306)
    value = (date_time.split(" ")[0]).split(":")[:3]
    value.append(date_time)
    #print(value)
    return value

def gpsInfo(img):
    # Get the data from image file and return a dictionary
    data = gpsphoto.getGPSData(img)
    #print(data["Latitude"], data["Longitude"])
    return str(data["Latitude"]) + ", " + str(data["Longitude"])

def getLocationDetails(strLnL):

   geolocator = Nominatim(user_agent="zesha")
   location = geolocator.reverse(strLnL)
   #print(location.address)
   return location.address

def getMetadata(img):
  res = getDateTime(img)
  res.append(getLocationDetails(gpsInfo(vf)))
  return res
    
 
if __name__ =="__main__":
    vf = sys.argv[1]
    extract(vf)
    #getLocationDetails(gpsInfo(vf))
    #getDateTime(vf)
    print(getMetadata(vf))