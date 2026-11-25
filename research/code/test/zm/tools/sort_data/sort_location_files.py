import pandas as pd
from pathlib import Path


'''
name - not na, or empty
state - not na, two char length
country - not na,
lat - between -80 and 80 and not null
lon - between -180 and 180 and not null

# 1. Check allowed set of values
valid_status = df["status"].isin(["active", "pending", "closed"])

# 2. Check numeric range
valid_age = df["age"].between(0, 120)

# 3. Check non-null & pattern match
valid_email = df["email"].notna() & df["email"].str.contains(r"@", na=False)

# Verify if entire DataFrame passes
is_valid = (valid_status & valid_age & valid_email).all()

# Extract only the rows that failed validation
invalid_rows = df[~(valid_status & valid_age & valid_email)]

'''
def validate_lat_lon(df):
    # Convert to numeric (non-numeric values become NaN)
    lat = pd.to_numeric(df['latitude'], errors='coerce')
    lon = pd.to_numeric(df['longitude'], errors='coerce')
    nm = df['name'].astype(str)
    st = df['state'].astype(str)
    cnt = df['country'].astype(str)

    # Create a boolean mask for valid coordinates
    not_null_island = ~((lat == 0) & (lon == 0))
    #df['is_valid'] = lat.between(-90, 90) & lon.between(-180, 180) & not_null_island

    valid_lat_lon = lat.between(-90, 90) & lon.between(-180, 180) & not_null_island

    valid_name = nm.notna()

    valid_state = st.notna()

    valid_country = cnt.notna()

    df['is_valid'] = (valid_lat_lon & valid_name & valid_state & valid_country).all()

    #df_invalid = df[~(is_valid)]
   
    # # Keep only valid rows
    valid_df = df[df['is_valid']].copy()

    # # Inspect invalid rows to see what went wrong
    invalid_df = df[~df['is_valid']]

    #df.drop(columns=['is_valid'], inplace=True)

    print(invalid_df)

    #print(df_invalid)

def sort_locations(loc_folder):

  folder = Path(loc_folder)

  # Get all CSV files in the folder (non-recursive)
  csv_files = list(folder.glob("*.csv"))

  for i, lfile in enumerate(csv_files):

        df = pd.read_csv(lfile, names=['name', 'state', 'country', 'latitude', 'longitude'])

        print(f"{i + 1}. Sorting a location file: {lfile.name} with \n {df.count()} locations \n")

        validate_lat_lon(df)

        df_sorted = df.sort_values(by=['country', 'state', 'name'], ascending=[False, True,False])

        df_sorted.drop(columns=['is_valid'], inplace=True)

        df_sorted.to_csv(lfile.name, header=False, index=False)


if __name__=="__main__":
    sort_locations("/users/bhal/work/vision/research/code/test/zm/schema/base_data/locations")
