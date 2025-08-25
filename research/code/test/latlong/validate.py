import pandas as pd

def validate_static_metadata_location(loc_file):
    df = pd.read_csv(loc_file)

    df.columns = ["name", "state", "country", "latitude", "longitude"]

    # print(df.head(80))

    #    print(f'find null values per column: {df.isnull()}')

    #    print(df['name'].isnull().sum())

    #    print(df.dtypes)

    #df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    df["latitude"] = pd.to_numeric(df["latitude"], errors='coerce')
    df['latitude'].to_csv('out1.csv')
    # print(df.isnull().sum(axis=1))

    print(df[df.isnull().any(axis=1)])

    #print(df[df["latitude"]])

if __name__=='__main__':
   #validate_static_metadata_location('/home/madhekar/work/vision/research/code/test/zm/schema/base_data/locations/in-cities.csv')   
   #validate_static_metadata_location('/home/madhekar/work/vision/research/code/test/zm/schema/base_data/locations/us-nationalparks.csv') 
   validate_static_metadata_location('/home/madhekar/work/home-media-app/data/app-data/static-metadata/locations/user-specific/madhekar/user-specific.csv')