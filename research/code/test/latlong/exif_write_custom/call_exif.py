from ExifTool_helper import ExifTool

fs = [
    "/home/madhekar/temp/faces/Anjali/Anjali60.png",
    "/home/madhekar/temp/faces/Anjali/Anjali61.png",
    "/home/madhekar/temp/faces/Anjali/Anjali63.png"
]

with ExifTool() as et:
    et.get_metadata(*fs)