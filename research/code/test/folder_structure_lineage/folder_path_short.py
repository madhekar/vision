import uuid
import os
import hashlib

def create_uuid_from_string(val: str):
   hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
   return uuid.UUID(hex=hex_string)

def path_encode(spath):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, spath))


# def path_encode_1(spath):
#   return str(uuid.uuid5(str.encode('zesha llc'),spath))

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

# print({x: path_encode(x) for x in plist})

# print({x: path_encode_1(x) for x in plist})


print({x: str(create_uuid_from_string(x)) for x in plist})
