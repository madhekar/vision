import pandas as pd

def sort_locations(loc_file):

    print(f"sorting a location file: {loc_file}")

    df = pd.read_csv(loc_file, names=['name', 'state', 'country', 'latitude', 'longitude'])

    print(df)

    df_sorted = df.sort_values(by=['country', 'state', 'name'], ascending=[False, True,False])

    df_sorted.to_csv('default_sorted.csv', header=False, index=False)