import os
import pandas as pd
import streamlit as st
import chromadb as cdb
from chromadb.config import Settings, DEFAULT_TENANT
from streamlit_dimensions import st_dimensions
import altair as alt
from utils.util import storage_stat as ss
from utils.config_util import config
from utils.util import model_util as mu


# https://www.color-hex.com/color-palette/164
colors = ['#6d765b','#A5BFA6']#['#847577','#cfd2cd']#['#f07162','#0081a7']#['#f97171','#8ad6cc']
#["#ae5a41", "#1b85b8"]#["#636B2F","#BAC095"] #["#9EB8A0", "#58855c"]#['#58855c','#0D3311']#["#BAC095", "#636B2F"]
#[ , , ,  ]

disc_usage_colors = ['#16697A', '#76B042', '#C23B22']
data_modality_colors = ['#16697A', '#76B042', '#C23B22','#FFA62B']


'''
altair==5.5.0
chromadb==0.5.3  --0.6.3
#country_converter==1.3.2
#fastparquet==2025.12.0
geopy==2.4.1
GPSPhoto==2.2.3
matplotlib==3.10.5 #3.10.8
numpy==1.26.4
pandas==2.2.2
piexif==1.1.3
Pillow==10.1.0   #12.1.1
PyYAML==6.0.3
scikit_learn==1.5.0

# streamlit==1.55.0
# streamlit_dimensions==0.0.1
# streamlit_extras==0.7.8
# streamlit_image_select==0.6.0

#
streamlit==1.39.0
streamlit-aggrid==1.2.1.post2
streamlit-avatar==0.1.3
streamlit-camera-input-live==0.2.0
streamlit-card==1.0.2
streamlit-dimensions==0.0.1
streamlit-embedcode==0.1.2
streamlit-extras==0.6.0
streamlit-image-coordinates==0.1.9
streamlit-image-select==0.6.0
streamlit-keyup==0.3.0
streamlit-option-menu==0.4.0
streamlit-toggle-switch==1.0.2
streamlit-tree-select==0.0.5
streamlit-vertical-slider==2.5.5
streamlit_faker==0.0.4
streamlit_folium==0.22.1
streamlit_imagegrid==0.0.7

#
torch==2.2.2
open-clip-torch==2.24.0
#tqdm==4.67.3

app launcher linux mint
-----------------------
ok changes .local python3 libraries
[Desktop Entry]
Name=zMedia
Exec=/home/madhekar/.local/bin/zmedia\n
Comment=zesha media browser application
Terminal=true
PrefersNonDefaultGPU=true
Icon=cinnamon-panel-launcher
Type=Application

good - isolated python3 libraries
[Desktop Entry]
Name=zm
Exec=./.zmedia/bin/zmedia
Comment=zmedia app launcher
Terminal=true
PrefersNonDefaultGPU=true
Icon=cinnamon-panel-launcher
Type=Application
'''

def extract_folder_paths():
    (
    data_path, 
    raw_data_path, 
    vectordb_path,
    image_collection_name,
    video_collection_name,
    text_collection_name,
    audio_collection_name,
    final_image_data_path, 
    final_video_data_path, 
    final_audio_data_path, 
    final_text_data_path
    ) = (
        config.overview_config_load()
    )
    ovr_path_list = [final_image_data_path, final_video_data_path, final_text_data_path, final_audio_data_path]
    vdb_list = [vectordb_path, image_collection_name, text_collection_name, video_collection_name,  audio_collection_name]

    return (data_path, raw_data_path, vdb_list, ovr_path_list)


def get_vdb_connection(vdb_path):
    client = None
    try:
      client = cdb.PersistentClient(vdb_path, tenant=DEFAULT_TENANT, settings=Settings(allow_reset=False))
      cc = client.count_collections()
      print(f"number of collections found: {cc}")  
    except Exception as e:
        print(f"exception occurred getting vdb client connection: {e}")  
    return client , cc   

def get_collection_record_count(vdb_list):

    vdb_path =  vdb_list.pop(0)

    print(vdb_list)

    client, num_collections = get_vdb_connection(vdb_path)

    collection_count = []
   
    for c in vdb_list: 
        try:
            cc = client.get_collection(name=c)
            val = cc.count()
            collection_count.append({"modality": c.removeprefix("multimodal_collection_"), 'count': val})
        except Exception as e:
            print(f"exception occurred while getting collection: {c} as: {e}")    
            collection_count.append({"modality": c.removeprefix("multimodal_collection_"), 'count': 0})
            continue
    print(collection_count)    
    return  collection_count, num_collections    


def disc_usage(tm, um, fm,w):

    v = w['width'] if w is not None else 1000
    mem = pd.DataFrame({"disc": ["Total", "Used", "Free"], "size": [tm, um, fm]})

    # This formats the value as an integer for cleaner presentation in the legend/tooltip
    mem["legend_label"] = (mem["disc"] + ":" + mem["size"].astype(str) + "GB")

    # Encode theta by the value, and color by the new combined label
    base = alt.Chart(mem).encode(
        theta=alt.Theta("size:Q").stack(True),
        radius=alt.Radius("size").scale(type="log", zero=True),
        color=alt.Color("disc:N", legend= None, scale=alt.Scale( 
                domain=["Total", "Used", "Free"],
                range=disc_usage_colors)) 
        )
    
   # 4. Create the pie (arc) layer innerRadius=int(0.05 * v), outerRadius=int(0.2 * v)
    pie = base.mark_arc(opacity=0.7, innerRadius=int(.02 * v), outerRadius=int(.021 * v), stroke="#fff", cornerRadius=3, strokeWidth=1).encode(
        tooltip=["disc:N", "size:Q", alt.Tooltip("legend_label:N")],
    )
    text = base.mark_text(align='center', radiusOffset=10, color="black").encode(text="size:Q")
    final_layer = alt.layer(pie , text)
    final_chart =  final_layer.properties(
        title="Disc Usage"
    ).configure_axisTop(
        labelPadding=50
    )
    st.altair_chart(final_chart, use_container_width=True)


def display_storage_metrics(tm, um, fm, ld, cc):
    c0, c1, c2, c3 = st.columns([.2, 1, 1, .2], gap="large", vertical_alignment="center")

    with c1:   
        #st.markdown("<div style='text-align: center;'> Disk Usage </div>", unsafe_allow_html=True)
        width = st_dimensions(key="c1_width")
        print(f"width {width}")
        disc_usage(tm, um, fm , width)

    with c2:
        st.markdown("<div style='text-align: center;'> Data per Modality </div>", unsafe_allow_html=True)
        df = pd.DataFrame(ld)
        print(df)
        base = alt.Chart(df).encode(
            x='modality:N',
            y="count:Q",
            text='count',
            color=alt.Color('modality:N', scale=alt.Scale(
                domain=["images", "videos", "texts", "audios"],
                range=data_modality_colors)) 
        )
        bar = base.mark_bar()
        text = base.mark_text(align='center', fontWeight="bold", dy=-10)
        ch =  alt.layer(bar , text)
        final_chart = ch.configure_axisTop(
            labelPadding=50
        ).properties(
            title="Data/ Modality"
        )
        st.altair_chart(final_chart, use_container_width=True)


def execute():

        data_path, final_data_path, vdb_list, opl = extract_folder_paths() 

        ld, cc = get_collection_record_count(vdb_list=vdb_list)
    
        #cc_data = pd.DataFrame({'collectons' : [cc]})
        efs = ss.extract_user_raw_data_folders(final_data_path)

        st.sidebar.markdown('##### :blue[**DATA SOURCES**]')
        for ds in efs:
                st.sidebar.write(f'**{ds}**')
        st.sidebar.divider()        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)   
        
        st.divider()

        display_storage_metrics(*ss.extract_server_stats(), ld, cc) #, dfi, dff)


if __name__ == "__main__":
    execute()