import json
import pandas as pd



def load_metadata():
  data = []
  with open("data.json", mode="r") as f:
    for line in f:
        data.append(json.loads(line))

    df =  pd.DataFrame(data)
  return df

def populate_vectordb(df):   
  df_id = df["id"]
  df_metadata = df[["timestamp","lat","lon","loc","nam","txt"]].T.to_dict().values()
  df_url = df["url"]

  print(f'id: \n {df_id.head()} \n metadata: \n {df_metadata} \n url: \n {df_url.head()} ')

if __name__=='__main__':

   df_vector_database = load_metadata()

   populate_vectordb()