import pandas as pd
def format_lat_lon(df):
    ll = ['GPSLatitude', 'GPSLongitude']
    df[ll] = df[ll].applymap(lambda x: str(round(float(x), 6)) if not x == '-' else x)

    print(df.head())

df = pd.read_csv('ll.csv')    