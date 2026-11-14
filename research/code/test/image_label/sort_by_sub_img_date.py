from pathlib import Path
import pandas as pd
import os

def sort_subfolders_by_min_date(
    df: pd.DataFrame,
    path_col: str = 'SourceFile',
    gpslat_col: str = 'GPSLatitude',
    gpslan_col: str = 'GPSLongitude',
    date_col: str = 'DateTimeOriginal',
    user_col:str = 'UserComment'
) -> pd.DataFrame:
  """Groups images by their immediate parent subfolder and returns a DataFrame

    of subfolders sorted by the earliest image creation date.
    """
  # Work on a copy to avoid mutating the original DataFrame
  temp_df = df.copy()

  #temp_df = temp_df[['SourceFile', 'DateTimeOriginal']]

  print(temp_df)

  # Ensure the date column is in datetime format for accurate chronological sorting
  temp_df[date_col] = pd.to_datetime(temp_df[date_col], format='%Y:%m:%d %H:%M:%S')

  #import pandas as pd

# Works on a single string or a DataFrame column
#df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y:%m:%d %H:%M:%S')


  # Extract the immediate parent folder name (e.g., '/root/sub_a/img.jpg' -> 'sub_a')
  temp_df['subfolder'] = temp_df[path_col].apply(lambda p: Path(p).parent.name)

  # Group by subfolder, find the minimum date, and sort ascending
  sorted_df = (
      temp_df.groupby('subfolder', as_index=False)[date_col]
      .min()
      .rename(columns={date_col: 'min_created_date'})
      .sort_values(by='min_created_date', ascending=False)
      .reset_index(drop=True)
  )

  return sorted_df

data = {
    'image_path': [
        '/photos/trip_b/img1.jpg',
        '/photos/trip_b/img2.jpg',
        '/photos/trip_a/img3.jpg',
        '/photos/trip_a/img4.jpg',
        '/photos/trip_c/img5.jpg',
    ],
    'created_date': [
        '2025-05-10 14:00:00',
        '2025-05-08 09:30:00',  # Earliest in trip_b
        '2025-01-15 12:00:00',  # Earliest in trip_a
        '2025-01-20 18:00:00',
        '2025-03-01 08:15:00',  # Earliest in trip_c
    ],
}

df = pd.read_csv ("/mnt/zmdata/home-media-app/data/input-data/error/img/missing-data/KIDDO/missing-metadata-wip.csv", dtype=str, usecols=['SourceFile','Idx','GPSLatitude','GPSLongitude','DateTimeOriginal','UserComment']) #('metadata.csv')
df = df.drop(['Idx'], axis=1)
#df.set_index("SourceFile", inplace=True)
#df = pd.DataFrame("/mnt/zmdata/home-media-app/data/input-data/error/img/missing-data/KIDDO/missing-metadata-wip.csv")
print(df.head(1))
result = sort_subfolders_by_min_date(df)
print(result)