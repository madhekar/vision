import glob
import shutil
import os

def getRecursive(rootDir):
    f_list = []
    for fn in glob.iglob(rootDir + "/**/*", recursive=True):
        if not os.path.isdir(os.path.abspath(fn)):
            f_list.append(
                (
                    str(os.path.abspath(fn)).replace(str(os.path.basename(fn)), ""),
                    os.path.basename(fn),
                )
            )
    return f_list


def getRecursive1(rootDir, types):
    f_list = []
    for t in types:
      for fn in glob.iglob(rootDir + "/**/"+t, recursive=True):
        print(os.path.abspath(fn))
        if fn is not None:
            f_list.append(
                (
                    str(os.path.abspath(fn)).replace(str(os.path.basename(fn)), ""),
                    os.path.basename(fn),
                )
            )
      
    return f_list

def get_files_by_types(rootDir, types):
    files_accumulator = []
    for files in types:
        files_accumulator.extend(glob.glob(rootDir + files, recursive=True))
    return files_accumulator    

def move_imges(src, tar, pattern):
    files = glob.glob(src + '/**/*.png')
    for file in files:
        fn = os.path.basename(file)
        shutil.move(file, os.path.join(tar , fn))


#fs = get_files_by_types('/home/madhekar/work/home-media-app/data/input-data/img', ['/**/*.jpg', '/**/*.png', '/**/*.jpeg'])

#fs = getRecursive('/home/madhekar/work/home-media-app/data/input-data/img')

#fs = getRecursive1('/home/madhekar/work/home-media-app/data/raw-data',['*.jpg', '*.png', '*.jpeg'])

move_imges(
    "/home/madhekar/work/home-media-app/data/raw-data",
    "/home/madhekar/work/home-media-app/data/input-data/img",
    '/**/*.png'
)

#print([print(f) for f in fs if f[1] is not None])