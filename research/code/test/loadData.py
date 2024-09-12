import json
import pandas as pd

data = []
with open("data.json", mode="r") as f:
    for line in f:
        data.append(json.loads(line))

#print(type(data[0]))

#print(data[0]["loc"])

df =  pd.DataFrame(data)

#print(df.head())

d_id = df["id"]
d_metadata = df[["timestamp","lat","lon","loc","nam","txt"]].T.to_dict().values()
d_url = df["url"]

print(f'id: \n {d_id.head()} \n metadata: \n {d_metadata} \n url: \n {d_url.head()} ')

