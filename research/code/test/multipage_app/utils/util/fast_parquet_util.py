import glob
import os.path
import pandas as pd
from fastparquet import write, ParquetFile
import fastparquet as fp
import country_converter as coco
from utils.util import us_states as ust
import streamlit as st
cc = coco.CountryConverter()


def transform_raw_locations(fpath):
    rdf = None
    try:
        with open(fpath, "r") as temp_f:
            f_arr = []
            for line in temp_f.readlines():
                a_ = line.split(",")
                a_ = [s.strip() for s in a_]
                if len(a_) > 5:
                    a_[0 : len(a_) - 4] = ["-".join(a_[0 : len(a_) - 4])]
                f_arr.append(a_)

            # create data frame
            df = pd.DataFrame(
                f_arr, columns=["name", "state", "country", "latitude", "longitude"]
            )

            # format country codes
            df["country"] = cc.pandas_convert(series=df["country"], to="ISO2")

            # standardize us state codes
            df["state"] = df["state"].apply(
                lambda x: ust.multiple_replace(ust.statename_to_abbr, x)
            )

            # overrite existing file in-place
            df.to_csv(fpath, sep=",", header=False, index=False)

            rdf = df
            print(f"+++ {fpath}: {rdf.shape}")
            st.info(f"{fpath}: {rdf.shape}")
    except Exception as e:
        st.error(f"failed to transform raw locations with exception: {e}")
    return rdf


def create_or_append_locations(raw_file, pfile_path):
    try:
        st.info(f"adding file: {raw_file} to parquat store: {pfile_path}")
        df = transform_raw_locations(raw_file)

        if not os.path.isfile(pfile_path):
            print(pfile_path)
            write(pfile_path, df)
        else:
            write(pfile_path, df, append=True)
    except Exception as e:
        st.error(f"create append locations parquet failed with exception: {e}")


def create_or_append_parquet(df, pfile_path):
    try:
        st.info(f"adding file: dataframe to parquat store: {pfile_path}")

        if not os.path.isfile(pfile_path):
            print(pfile_path)
            write(pfile_path, df)
        else:
            write(pfile_path, df, append=True)
    except Exception as e:
        st.error(f"create append locations parquet failed with exception: {e}")


def parquet_generator(pfile_path, chunk_size=10):
    pf = ParquetFile(pfile_path)   
    for i in range(0, len(pf), chunk_size):
        yield pf.to_pandas(rows=(i, min(i + chunk_size. len(pf))))     

def read_parquet_file(file_path):
    rdf = None
    try:
        pf = fp.ParquetFile(file_path)
        rdf = pf.to_pandas()
        rdf['name'].str.strip()
    except Exception as e:
        st.error(f"get all locations failed with exception: {e}")
    return rdf

def get_all_loc_by_country(file_path, country):
    rdf = None
    try:
        pf = fp.ParquetFile(file_path)
        rdf = pf.to_pandas(filters=[("country", "==", country)], row_filter=True)
    except Exception as e:
        st.error(f"get all locations by contry: {country} failed with exception: {e}")
    return rdf

def get_all_loc_by_country_and_state(file_path, country, state):
    rdf = None
    try:
        pf = fp.ParquetFile(file_path)
        df_ = pf.to_pandas(filters=[("country", "=", country)], row_filter=True)
        rdf = df_[df_["state"] == state]
    except Exception as e:
        st.error(f"get all locations by country: {country} and state: {state} failed with exception: {e}")
    return rdf

def combine_all_default_locations(location_root):
    try:
        locations_file_list = glob.glob(os.path.join(location_root, "*.csv"))
        pdf = []
        for f in locations_file_list:
            print(f'>> {f}')
            pdf.append(transform_raw_locations(f))
        df_comb = pd.concat(pdf, ignore_index=True)  
        print(f'>>{len(df_comb)}')  
    except Exception as e:
        st.error(
            f"create append locations parquet for file: {f} failed with exception: {e}"
        )
    return df_comb

def add_all_locations(location_root, user_location_root, user_location_metadata_file, parquet_file_path):
    try:
        locations_file_list = glob.glob(os.path.join(location_root, "*.csv"))
        for f in locations_file_list:
            create_or_append_locations(f, parquet_file_path)
    except Exception as e:
        st.error(f"create append locations parquet for file: {f} failed with exception: {e}")
   
    try:
        # locations_file_list = glob.glob(os.path.join(user_location_root, user_location_metadata_file))
        # for f in locations_file_list:
        create_or_append_locations(os.path.join(user_location_root, user_location_metadata_file), parquet_file_path)
    except Exception as e:
        st.error(f"create append locations parquet for file: {f} failed with exception: {e}")


def init_location_cache(parquet_file_path):
    
    df = read_parquet_file(parquet_file_path)   

    df["LatLon"] = df[["latitude", "longitude"]].apply(tuple, axis=1)

    df.drop(columns=["latitude", "longitude"], inplace=True)
    df.head(10)
    return df     


if __name__ == "__main__":
    parquet_file_path = "parquet/static_locations.parquet"

    # add_all_locations('locations/', parquet_file_path=parquet_file_path)

    # rdf = read_parquet_file(file_path=parquet_file_path)
    # print(rdf.head())

    # rdf = get_all_loc_by_country(parquet_file_path, "us")
    # print(rdf.head())

    rdf = get_all_loc_by_country_and_state(parquet_file_path, "US", "CA")
    print(f"results with {rdf.shape[0]} rows and  {rdf}")
