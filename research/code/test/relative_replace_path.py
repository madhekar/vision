import os

o_path ='/home/madhekar/work/home-media-app/data/input-data/img/ca-sanFrancisco/IMG_1578.JPG'

r_path = os.path.relpath(o_path, "/home/madhekar/work/home-media-app/data/input-data/img")

n_path = os.path.join("/home/madhekar/work/home-media-app/data/final-data/img/", r_path)

print(f"{o_path} \n {r_path} \n {n_path}")

f_path = o_path.replace(
    "/home/madhekar/work/home-media-app/data/input-data/img",
    "/home/madhekar/work/home-media-app/data/final-data/img",
)

print(f_path)