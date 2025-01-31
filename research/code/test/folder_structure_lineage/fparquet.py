import glob
import os.path
import pandas as pd
from fastparquet import write
import fastparquet as fp
import country_converter as coco
import us_states as ust

cc = coco.CountryConverter()

def transform_raw_locations(fpath):
    with open(fpath, "r") as temp_f:
        f_arr = []
        for line in temp_f.readlines():
            a_ = line.split(",")
            a_ = [s.strip() for s in a_]
            if len(a_) > 5:
                a_[0 : len(a_) - 4] = ["-".join(a_[0 : len(a_) - 4])]
            f_arr.append(a_)
        # create data frame
        df = pd.DataFrame(f_arr, columns=["name", "state", "country", "latitude", "longitude"])

        # format country codes
        df["country"] = cc.pandas_convert(series=df["country"], to="ISO2")
        # standardize us state codes
        df["state"] = df["state"].apply(
            lambda x: ust.multiple_replace(ust.statename_to_abbr, x)
        )
        df.to_csv(fpath, sep=',', header=False, index=False)
        return df

def create_append_locations(raw_file, pfile_path):
    try:
        print(f"adding file: {raw_file} to parquat store: {pfile_path}")
        df = transform_raw_locations(raw_file) 

        if not os.path.isfile(pfile_path):
           print(pfile_path)
           write(pfile_path, df)
        else:
           write(pfile_path, df, append=True) 
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

def add_all_locations(location_root, parquet_file_path):
    try:
      locations_file_list = glob.glob(os.path.join(location_root, '*.csv'))
      for f in locations_file_list:
         create_append_locations(f, parquet_file_path)
    except Exception as e:
        print(f"create append locations parquet for file: {f} failed with exception: {e}")

if __name__=='__main__':
    parquet_file_path = 'parquet/static_locations.parquet'
 
    #add_all_locations('locations/', parquet_file_path=parquet_file_path)

    # rdf = read_parquet_file(file_path=parquet_file_path)
    # print(rdf.head())

    # rdf = get_all_loc_by_country(parquet_file_path, "us")
    # print(rdf.head())

    rdf = get_all_loc_by_country_and_state(parquet_file_path, "IN", "MH")
    print(f"results with {rdf.shape[0]} rows and  {rdf}")
