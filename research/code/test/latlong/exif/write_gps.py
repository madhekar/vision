import subprocess
from GPSPhoto import gpsphoto

def set_gps_data(img_path, lat, lon):
    try:
        print(f"----> {img_path} {lat} {lon}")
        command = [
            "/usr/bin/exiftool",
            "-m",
            "-F",
            f"-GPSLatitude={str(lat)}",
            f"-GPSLatitudeRef={'N' if lat >= 0 else 'S'}",
            f"-GPSLongitude={str(lon)}",
            f"-GPSLongitudeRef={'E' if lon >= 0 else 'W'}",
            "-overwrite_original",  # Overwrite the original file to avoid duplication
            img_path
        ]
        subprocess.run(command, check=True)
    except Exception as e:
        print(f"Exception: {e} while setting gps data for {img_path} ")


def get_gps_data(img_path):
    gps = ()
    try:
        # Get the data from image file and return a dictionary
        data = gpsphoto.getGPSData(img_path)

        if "Latitude" in data and "Longitude" in data:
            gps = (round(data["Latitude"],6), round(data["Longitude"], 6))
    except Exception as e:
        print(f'exception occurred in extracting lat/ lon data: {e}')
    return gps


img = "/mnt/zmdata/home-media-app/data/input-data/img/Samsung USB/27c7a2a1-5138-594b-adb7-65a6888680b5/IMG_0157.jpg"
set_gps_data(img, 32.968689, -117.184243)   

print(get_gps_data(img))