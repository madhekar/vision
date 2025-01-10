import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

def define_schema():
    static_location_schema = pa.schema([
    ('country', pa.string()),
    ('state', pa.string()),
    ('name', pa.int()),
    ('description', pa.string()),
    ('latitude', pa.float32()),
    ('longitude', pa.float32())
    ])

    df = pd.read_csv("default-locations.csv")
    #create default table
    table = pa.Table.from_dataframe(df, schema=static_location_schema) 
    try:
      pq.write_table(table, 'pqt/static_locations.parquet')
    except Exception as e:
       print(f'failed with exception: {e}')