
import os
import pandas as pd

df = pd.read_csv('locations.csv', index_col=0)

print(df.head())

d = df.to_dict("split")
d = dict(zip(d["index"], d["data"]))

print(d)

d["ca-science"] = ["esha science fields day",0.0,-0.1]

df = pd.DataFrame.from_dict(d, orient="index")

print(df.head())
