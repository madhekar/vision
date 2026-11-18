import pandas as pd

n_file =  "/mnt/zmdata/home-media-app/data/app-data/static-metadata/locations/default/us-nationalparks.csv"
df = pd.read_csv(n_file)

print(df)

