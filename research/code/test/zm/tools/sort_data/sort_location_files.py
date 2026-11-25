import pandas as pd
from pathlib import Path

def validate_lat_lon(df):
    # Convert to numeric (non-numeric values become NaN)
    lat = pd.to_numeric(df['latitude'], errors='coerce')
    lon = pd.to_numeric(df['longitude'], errors='coerce')

    # Create a boolean mask for valid coordinates
    not_null_island = ~((lat == 0) & (lon == 0))
    df['is_valid'] = lat.between(-90, 90) & lon.between(-180, 180) & not_null_island

    # Keep only valid rows
    valid_df = df[df['is_valid']].copy()

    # Inspect invalid rows to see what went wrong
    invalid_df = df[~df['is_valid']]

    #df.drop(columns=['is_valid'], inplace=True)

    print(invalid_df)

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
