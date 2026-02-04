import pandas as pd
c_names = ["lib_name"]
df = pd.read_csv("req.txt" ,header=None, names= c_names)
print(df.head(10))

lst = df['lib_name'].to_list()

print(lst)