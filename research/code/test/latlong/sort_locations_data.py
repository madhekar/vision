import pandas as pd

def sort_locations(default_loc_file):
    df = pd.read_csv(default_loc_file, names=['name', 'state', 'country', "latitude", 'longitude'])

    df_sorted = df.sort_values(by=['country', 'state'], ascending=[True, False])

    df_sorted.to_csv('default_sorted.csv', header=False, index=False)

sort_locations("/users/bhal/work/vision/research/code/test/zm/schema/base_data/locations/default.csv")    