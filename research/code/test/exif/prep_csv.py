import os
import csv
import pandas as pd
#/Users/bhal/work/vision/research/code/test/zm/schema/base_data/locations/default.csv
def format_loc_data(file_path):
    df = pd.read_csv(file_path, delimiter=',')

    print(df)
    # df_sorted = df.sort_values(by=['', 'column_B'])

    # pt = df_sorted.pivot_table()

    # print(pt)

format_loc_data("./default.csv")
