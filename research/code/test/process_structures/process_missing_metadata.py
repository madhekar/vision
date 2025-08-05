import pandas as pd
import os

fo = "/home/madhekar/work/home-media-app/data/input-data/error/img/missing-data/AnjaliBackup/"
fn = 'missing-metadata-wip.csv'

df = pd.read_csv(os.path.join(fo,fn))

fc = len(df)

dfm = df[(df["GPSLatitude"] == "-") | (df["DateTimeOriginal"] == "-")]

fmc = len(dfm)

print(f'full count {fc} missing metadata count {fmc}')

print(dfm.head(20))