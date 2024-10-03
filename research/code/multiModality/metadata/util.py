import yaml
import pyexiv2
import streamlit as st

@st.cache_resource
def config_load():
    with open("metadata_conf.yaml") as prop:
        dict = yaml.safe_load(prop)

        print(
            "* * * * * * * * * * * Metadata Generator Properties * * * * * * * * * * * *"
        )
        print(dict)
        print(
            "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"
        )
        static_metadata_path = dict["metadata"]["static_metadata_path"]
        static_metadata_file = dict["metadata"]["static_metadata_file"]
        missing_metadata_path = dict["metadata"]["missing_metadata_path"]
        missing_metadata_file = dict["metadata"]["missing_metadata_file"]
    return (
        static_metadata_path,
        static_metadata_file,
        missing_metadata_path,
        missing_metadata_file,
    )

def to_deg(value, loc):
    if value < 0:
        loc_value = loc[0]
    elif value > 0:
        loc_value = loc[1]
    else:
        loc_value = ""
    abs_value = abs(value)
    deg = int(abs_value)
    t1 = (abs_value - deg) * 60
    min = int(t1)
    sec = round((t1 - min) * 60, 5)
    return (deg, min, sec, loc_value)


def setGpsLocation(fname, lat, lng):
    lat_deg = to_deg(lat, ["S", "N"])
    lng_deg = to_deg(lng, ["W", "E"])

    print("lat:", lat_deg, " lng:", lng_deg)

    # convert decimal coordinates into degrees, minutes and seconds
    exiv_lat = (
        pyexiv2.Rational(lat_deg[0] * 60 + lat_deg[1], 60),
        pyexiv2.Rational(lat_deg[2] * 100, 6000),
        pyexiv2.Rational(0, 1),
    )
    exiv_lng = (
        pyexiv2.Rational(lng_deg[0] * 60 + lng_deg[1], 60),
        pyexiv2.Rational(lng_deg[2] * 100, 6000),
        pyexiv2.Rational(0, 1),
    )

    exiv_image = pyexiv2.Image(fname)
    exiv_image.readMetadata()
    exif_keys = exiv_image.exifKeys()
    print("exif keys: ", exif_keys)

    exiv_image["Exif.GPSInfo.GPSLatitude"] = exiv_lat
    exiv_image["Exif.GPSInfo.GPSLatitudeRef"] = lat_deg[3]
    exiv_image["Exif.GPSInfo.GPSLongitude"] = exiv_lng
    exiv_image["Exif.GPSInfo.GPSLongitudeRef"] = lng_deg[3]
    exiv_image["Exif.Image.GPSTag"] = 654
    exiv_image["Exif.GPSInfo.GPSMapDatum"] = "WGS-84"
    exiv_image["Exif.GPSInfo.GPSVersionID"] = "2 0 0 0"

    exiv_image.writeMetadata()


def setDateTimeOriginal(fname, dt):
    exiv_image = pyexiv2.Image(filename=fname)
    exiv_image["Exif"][pyexiv2.ExifIFD.DateTimeOriginal] = dt
    exiv_image.writeMetaDate()