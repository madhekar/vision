import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


def define_schema():
    static_location_schema = pa.schema(
        [
            ("id", pa.int8()),
            ("country", pa.string()),
            ("state", pa.string()),
            ("desc", pa.string()),
            ("lat", pa.float64()),
            ("lon", pa.float64()),
        ]
    )
    return static_location_schema


def create_default_location(schema):
    df = pd.read_csv("default-locations.csv", delimiter=",")
    # create default table
    table = pa.Table.from_pandas(df, schema=schema)
    try:
        pq.write_table(
            table, "pqt/static_locations.parquet"
        )  # , partition_cols=['country','state'])
    except Exception as e:
        print(f"failed with exception: {e}")


def read_parquet_file():
    df = pd.read_parquet("pqt/")
    print(df.head())


def get_all_loc_by_country(country):
    df = pd.read_parquet("pqt/")
    print(df[df["country"] == country])


def get_all_loc_by_country_and_state(country, state):
    df = pd.read_parquet("pqt/")
    print(df[(df["country"] == country) & (df["state"] == state)])


create_default_location(define_schema())

# read_parquet_file()

get_all_loc_by_country("us")

get_all_loc_by_country_and_state("us", "california")
