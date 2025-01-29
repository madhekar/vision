
import os.path
import pandas as pd
from fastparquet import write
import fastparquet as fp

def create_append_locations(raw_file):
    try:
        df = pd.read_csv(raw_file, delimiter=",")
        df.columns = ['name','country','state','latitude','longitude']

        if not os.path.isfile(file_path):
           write(file_path, df)
        else:
           write(file_path, df, append=True) 
    except Exception as e:
        print(f"create append locations parquet failed with exception: {e}")


def read_parquet_file(file_path):
    rdf = None
    try:
      pf = fp.ParquetFile(file_path)
      rdf = pf.to_pandas()
    except Exception as e:
      print(f"get all locations failed with exception: {e}")
    return rdf


def get_all_loc_by_country(file_path, country):
    rdf = None
    try:
        pf = fp.ParquetFile(file_path)
        rdf = pf.to_pandas(filters=[('country', "==", country)], row_filter=True)
    except Exception as e:
        print(f"get all locations by contry: {country} failed with exception: {e}")
    return rdf


def get_all_loc_by_country_and_state(file_path, country, state):
    rdf = None
    try:
        pf = fp.ParquetFile(file_path)
        df_ = pf.to_pandas(filters=[("country", "=", country)], row_filter=True)
        rdf = df_[df_["state"] == state]
    except Exception as e:
        print(
            f"get all locations by country: {country} and state: {state} failed with exception: {e}"
        )
    return rdf


if __name__=='__main__':
    file_path = 'parquet/static_locations.parquet'
    raw_csv_path = "locations/default.csv"

    create_append_locations(raw_file=raw_csv_path)

    rdf = read_parquet_file(file_path=file_path)
    print(rdf.head())

    rdf = get_all_loc_by_country(file_path, "us")
    print(rdf.head())

    rdf = get_all_loc_by_country_and_state(file_path, "us", "California")
    print(rdf.head())