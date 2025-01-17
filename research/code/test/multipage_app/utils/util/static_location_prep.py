import os
from geopy.geocoders import Nominatim
import pyarrow as pa
import pyarrow.parquet as pq
import time
import pandas as pd
from pprint import pprint


def init(static_locaton_prep_data_path, static_locaton_prep_file):
    geolocator = Nominatim(user_agent="name_ofgent")
    df = pd.read_csv(
        os.path.join(static_locaton_prep_data_path, static_locaton_prep_file),
        delimiter=",",
    )
    pprint(df)
    return geolocator, df


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


def extract_location_details_with_error_chk(geolocator, df):
    loc_list = []
    i = 1
    for row in df.itertuples():
        d = {}
        try:
            loc = geolocator.geocode(
                row.address, exactly_one=True, timeout=60, addressdetails=True
            )
            data = loc.raw
            print(data)
            data = data["address"]
            state = country = ""
            if "state" in data:
                state = str(data["state"]).lower()
            else:
                state = "unknown"
            country = str(data["country_code"]).lower()
            d["id"] = i
            d["desc"] = row.name
            d["state"] = state
            d["country"] = country
            d["lat"] = loc.latitude
            d["lon"] = loc.longitude
            loc_list.append(d)
            i += 1
        except Exception as e:
            pprint(f"error: geocode failed for {row.address} with exception {e}")
        time.sleep(2)

    # create dataframe to return
    dfr = pd.DataFrame(
        loc_list, columns=["id", "country", "state", "desc", "lat", "lon"]
    ).set_index("id", drop=True)
    return dfr


def prepare_static_metadata_locations(static_locaton_prep_data_path, static_locaton_prep_data_file, static_location_path, static_location_file):
    try:
        #create geo locator and read addres list
        geo_locator, dfi = init(static_locaton_prep_data_path, static_locaton_prep_data_file)

        #define parquet file schema
        parquet_schema = define_schema()

        #extract location metadata such as lat, lon, country, state
        dfr = extract_location_details_with_error_chk(geolocator=geo_locator, df=dfi)

        #create parquet file with schema, and generated metadata
        create_default_location(static_location_path, static_location_file, parquet_schema, dfr)

    except Exception as e:
        pprint(f"error: exception in preparing static metadata: {e}")


def create_default_location(static_location_path, static_location_file, schema, df):
    try:
        table = pa.Table.from_pandas(df, schema=schema)
        pq.write_table(table, os.path.join(static_location_path, static_location_file))  # , partition_cols=['country','state'])
    except Exception as e:
        pprint(f"error: create parquet table failed with exception: {e}")


def read_parquet_file(static_location_path):
    df = pd.read_parquet(static_location_path)
    return df


def get_all_loc_by_country(country, static_location_path):
    df = pd.read_parquet(static_location_path)
    return df[df["country"] == country]


def get_all_loc_by_country_and_state(country, state, static_location_path):
    df = pd.read_parquet(static_location_path)
    return df[(df["country"] == country) & (df["state"] == state)]
