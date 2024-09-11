import json
import pandas as pd

data = []
with open("data.json", mode="r") as f:
    for line in f:
        data.append(json.loads(line))

print(type(data[0]))

print(data[0]["loc"])

df =  pd.DataFrame(data)

print(df["id"])