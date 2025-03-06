import os.path
import pandas as pd
from fastparquet import write
import fastparquet as fp

def read_parquet_file(file_path):
    rdf = None
    try:
        pf = fp.ParquetFile(file_path)
        rdf = pf.to_pandas()
    except Exception as e:
        print(f"get all locations failed with exception: {e}")
    return rdf

df = read_parquet_file(os.path.join("/home/madhekar/work/home-media-app/data/app-data/static-metadata","static_locations.parquet"))
df['LatLon'] = df[['latitude', 'longitude']].apply(tuple, axis=1)

df.drop(columns=['latitude','longitude', 'state', 'country'], inplace=True)

row = df.loc[df.LatLon == ("45.069157", "-83.437386")].values.flatten().tolist()
#row = df.loc[df["LatLon"] == ("45.069157", "-83.437386")]

print(row)

# df.drop(columns=['latitude','longitude', 'state', 'country'], inplace=True)
# dicts = df.to_dict(orient='records')

# # merge
# fd = {v:k for d in dicts for k, v in d.values()}

# for d in dicts:
#     lst = list(d.values())
#     print(lst[0], lst[1])
#     fd[lst[1]] = lst[0]


# print(fd)