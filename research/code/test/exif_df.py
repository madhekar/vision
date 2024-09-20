import pandas as pd

df = pd.read_csv('out.csv')

print('*------Image Files-------*')
pattern = 'jpg|jpeg|png'
mask = df['SourceFile'].str.contains(pattern, case=False, na=False)
print(df[mask])

dfi = df[mask]
print('*------Missing Date------*')
pattern = '-'
missing_date = dfi["DateTimeOriginal"].str.contains(pattern, case=False, na=False)
print(dfi[missing_date])

dfid = dfi[missing_date]
print("*------Missing Lat/Long------*")
pattern = "-"
missing_latlong= dfi["GPSLatitude"].str.contains(pattern, case=False, na=False)
print(dfi[missing_latlong])