import os
import csv
import pandas as pd
#/Users/bhal/work/vision/research/code/test/zm/schema/base_data/locations/default.csv
def format_loc_data(file_path):
    df = pd.read_csv(file_path, delimiter=',')

    df_sorted = df.sort_values(by=['country', 'state'], ascending=[False, True])

    print(df_sorted)

    df_sorted.to_csv("sorted_default.csv", index=False)

format_loc_data("./default.csv")
