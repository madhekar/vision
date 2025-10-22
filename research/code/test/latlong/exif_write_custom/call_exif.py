from ExifTool_helper import ExifTool

fs = [
    "/Users/emadhekar/Pictures/00e39dd1-e166-49ae-9f9e-e83b2546b056.JPG",
    "/Users/emadhekar/Pictures/1a5e9da6-462d-4f2f-a289-5c45e0db1176.JPG",
    "/Users/emadhekar/Pictures/5c726d80-e2db-4600-8483-f6c1b88fcec2.JPG",
]

with ExifTool() as et:
    et.get_metadata(*fs)