import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import os

def define_schema():
    static_location_schema = pa.schema([
    ('id', pa.int8()),
    ('name', pa.string()),
    ('country', pa.string()),
    ('state', pa.string()),
    ('latitude', pa.float64()),
    ('longitude', pa.float64())
    ])
    return static_location_schema

def create_default_location(schema):
    try:
        df = pd.read_csv("default-locations.csv", delimiter=',')
        #create default table
        table = pa.Table.from_pandas(df, schema=schema) 
        pq.write_table(table, 'pqt/static_locations.parquet')#, partition_cols=['country','state'])
    except Exception as e:
        print(f'create default locations parquet failed with exception: {e}')

   
def read_parquet_file():
   
   df = pd.read_parquet('pqt/')
   return df

def append_locations(df,path='pqt', file_name='default-locations.parquet'):
    try:
        df.to_parquet(os.path.join(path, file_name), engine="pyarrow", mode="append")
    except Exception as e:
        print(f"append locations failed with exception: {e}")
   
def get_all_loc_by_country(country):
    rdf = None
    try:
        df = pd.read_parquet("pqt/")
        rdf = df[(df["country"] == country)]
    except Exception as e:
        print(f"get all locations by contry: {country} failed with exception: {e}")
    return rdf

def get_all_loc_by_country_and_state(country, state):
    rdf = None
    try:
      df = pd.read_parquet("pqt/")
      rdf = df[(df["country"] == country) & (df["state"] == state)]
    except Exception as e:
       print(f'get all locations by contry: {country} and state: {state} failed with exception: {e}') 
    return rdf    

create_default_location(define_schema())   

read_parquet_file()

get_all_loc_by_country('us')

get_all_loc_by_country_and_state('us', 'california')