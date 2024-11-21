import uuid
import os

def path_encode(spath):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, spath))


plist = [
    "/home/madhekar/work/home-media-portal/data/raw-data/1",
    "/home/madhekar/work/home-media-portal/data/raw-data/5",
    "/home/madhekar/work/home-media-portal/data/raw-data/0",
    "/home/madhekar/work/home-media-portal/data/raw-data/",
    "/home/madhekar/work/home-media-portal/data/raw-data",
    "/home",
    "",
    "/"
]

print({x: path_encode(x) for x in plist})