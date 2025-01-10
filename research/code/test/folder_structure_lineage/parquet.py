import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

def define_schema():
    static_location_schema = pa.schema([
    ('country', pa.string()),
    ('state', pa.string()),
    ('name', pa.string()),
    ('description', pa.string()),
    ('latitude', pa.float32()),
    ('longitude', pa.float32())
    ])
    return static_location_schema

def create_default_location(schema):

    df = pd.read_csv("default-locations.csv")
    #create default table
    table = pa.Table.from_pandas(df, schema=schema) 
    try:
      pq.write_table(table, 'pqt/static_locations.parquet')
    except Exception as e:
       print(f'failed with exception: {e}')

def read_parquet_file():
   df = pd.read_parquet('pqt/')
   print(df.head())

def get_all_loc_by_country(country):
   df = pd.read_parquet('pqt/')   
   print(df[df['country'] == country])

def get_all_loc_by_country_and_state(country, state):
    df = pd.read_parquet("pqt/")
    print(df[(df["country"] == country) & (df["state"] == state)])

create_default_location(define_schema())   

#read_parquet_file()

get_all_loc_by_country('usa')

get_all_loc_by_country_and_state('usa', 'ca')