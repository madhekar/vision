from pathlib import Path
import pandas as pd


def sort_subfolders_by_min_date(
    df: pd.DataFrame,
    path_col: str = 'image_path',
    date_col: str = 'created_date',
) -> pd.DataFrame:
  """Groups images by their immediate parent subfolder and returns a DataFrame

    of subfolders sorted by the earliest image creation date.
    """
  # Work on a copy to avoid mutating the original DataFrame
  temp_df = df.copy()

  # Ensure the date column is in datetime format for accurate chronological sorting
  temp_df[date_col] = pd.to_datetime(temp_df[date_col])

  # Extract the immediate parent folder name (e.g., '/root/sub_a/img.jpg' -> 'sub_a')
  temp_df['subfolder'] = temp_df[path_col].apply(lambda p: Path(p).parent.name)

  # Group by subfolder, find the minimum date, and sort ascending
  sorted_df = (
      temp_df.groupby('subfolder', as_index=False)[date_col]
      .min()
      .rename(columns={date_col: 'min_created_date'})
      .sort_values(by='min_created_date', ascending=True)
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


df = pd.DataFrame(data)
print(df)
result = sort_subfolders_by_min_date(df)
print(result)