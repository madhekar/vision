import os
import glob
import pandas as pd

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


lst = getRecursive("/home/madhekar/work/home-media-app/data/input-data-1/img/AnjaliBackup")

df = pd.DataFrame(lst, columns=('path', 'file'))

print(df.head(20))