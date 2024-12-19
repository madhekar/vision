
import pandas as pd
import plotly.express as px
import streamlit as st


def extract_stats_of_metadata_file(metadata_path):
    print(metadata_path)
    mdf = pd.read_csv(metadata_path)
    print(mdf.head(10))
    clat = mdf[mdf['GPSLatitude'] == "-"].shape[0]
    clon = mdf[mdf['GPSLatitude'] == "-"].shape[0]
    cdatetime = mdf[mdf['DateTimeOriginal'] == "-"].shape[0]
    clatlong_n_datetime = mdf[(mdf['DateTimeOriginal'] == "-") & (mdf['GPSLatitude'] == "-")].shape[0]

    return {
        "lat": clat,
        #"lon" : clon,
        "datetime": cdatetime,
        #"latlon_n_datetime": clatlong_n_datetime
    }
    '''
    /home/madhekar/work/home-media-app/data/input-data/error/img/missing-data/missing-metadata-wip.csv
    '''

if __name__=='__main__':
       dict =  extract_stats_of_metadata_file('/home/madhekar/work/home-media-app/data/input-data/error/img/missing-data/missing-metadata-wip.csv')

       print(dict)

       df = pd.DataFrame.from_dict([dict])

       print(df)

       fig = px.pie(df, names=df.columns, values=df.values[0])
       st.plotly_chart(fig, use_container_width=True)