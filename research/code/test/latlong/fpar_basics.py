import pandas as pd
from fastparquet import write

data = {
    'col_str': ['apple', 'banana', 'cherry'],
    'col_int': [1, 2, 3],
    'col_json': [3.4,5.7,8.4],
    'col_bool': [True, False, True]
}
df = pd.DataFrame(data)

# Specify object_encoding for 'col_str' as utf8, 'col_json' as json, and 'col_bool' as bool
object_encodings = {
    'col_str': 'utf8',
    'col_int' : 'int32',
    'col_json': 'decimal',
    'col_bool': 'bool'
}

write(
    'output.parquet',
    df,
    object_encoding=object_encodings
)

print(df.dtypes)