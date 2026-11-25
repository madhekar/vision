import pandas as pd
from pathlib import Path

def sort_locations(loc_folder):

  folder = Path(loc_folder)

  # Get all CSV files in the folder (non-recursive)
  csv_files = list(folder.glob("*.csv"))

  for i, lfile in enumerate(csv_files):

        df = pd.read_csv(lfile, names=['name', 'state', 'country', 'latitude', 'longitude'])

        print(f"{i + 1}. Sorting a location file: {lfile.name} with \n {df.count()} locations \n")

        df_sorted = df.sort_values(by=['country', 'state', 'name'], ascending=[False, True,False])

        df_sorted.to_csv(lfile.name, header=False, index=False)


if __name__=="__main__":
    sort_locations("/users/bhal/work/vision/research/code/test/zm/schema/base_data/locations")
