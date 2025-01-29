# import pandas as pd
# from pathlib import Path

# df = pd.DataFrame({'col1': [1, 2,], 'col2': [3, 4]})
# file_path = Path("parquet/output.parquet")

# if file_path.exists():
#   df.to_parquet(file_path, engine='fastparquet', append=True)
# else:
#   df.to_parquet(file_path, engine='fastparquet')

import os.path
import pandas as pd
from fastparquet import write
import fastparquet as fp

df = pd.DataFrame(data={'col1': [1, 2,], 'col2': [3, 4]})
file_path = "parquet/write_parq_row_group.parquet"
if not os.path.isfile(file_path):
  write(file_path, df)
else:
  write(file_path, df, append=True)

df = fp.ParquetFile(file_path).to_pandas(filters=[('col1','==',1)], row_filter=True)

print(df)

# import pyarrow as pa
# import pyarrow.parquet as pq

# # Existing data
# table1 = pa.Table.from_pydict({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
# pq.write_table(table1, "my_data.parquet")

# # New data to append
# table2 = pa.Table.from_pydict({"col1": [4, 5], "col2": ["d", "e"]})

# # Append data
# pq.write_table(table2, "my_data.parquet", existing_data_behavior="append")