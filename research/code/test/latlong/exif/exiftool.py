from exiftool import ExifTool

files = ["/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/e8828925-b35c-4779-a62e-1adcb11a156c.jpg",
"/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/f2ab6bfb-5adf-495c-9518-470af9dce1cf.jpg", 
"/home/madhekar/work/home-media-app/data/input-data/img/Samsung_USB/2a98fafb-a921-519f-8561-ed25ccd997de/fce7616e-d485-403b-ba29-e33d5b80df09.jpg"]
with ExifTool() as et:
    metadata = et.get_metadata_batch(files)
for d in metadata:
    print("{:20.20} {:20.20}".format(d["SourceFile"],
                                     d["EXIF:DateTimeOriginal"]))