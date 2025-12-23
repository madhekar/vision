import os
from pathlib import Path
import pandas as pd

import streamlit as st
import altair as alt


def get_folder_metrics(folder_path):
    """
    Calculates the total file count and size (in bytes) for a given folder path.

    Args:
        folder_path (str or Path): The path to the folder.

    Returns:
        tuple: (file_count, total_size_bytes)
    """
    file_count = 0
    total_size_bytes = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Skip if it is a symbolic link to avoid issues
            if not os.path.islink(fp):
                try:
                    total_size_bytes += os.path.getsize(fp)
                    file_count += 1
                except OSError:
                    # Handle cases where file might be inaccessible
                    print(f"Error accessing file: {fp}")
                    pass
    return file_count, total_size_bytes

def bytes_to_gb(bytes_size):
    """Converts bytes to gigabytes."""
    return bytes_size / (1024**3)

def acquire_data():

    # Define the list of folders to process
    folders_to_check = [
        "/home/madhekar/work/home-media-app/data/input-data/img",
        "/home/madhekar/work/home-media-app/data/input-data/video",
        "/home/madhekar/work/home-media-app/data/input-data/txt",
        "/home/madhekar/work/home-media-app/data/input-data/audio",
        "/home/madhekar/work/home-media-app/data/input-data/error/img/duplicate",
        "/home/madhekar/work/home-media-app/data/input-data/error/img/missing-data",
        "/home/madhekar/work/home-media-app/data/input-data/error/img/quality",
        "/home/madhekar/work/home-media-app/data/input-data/error/video/duplicate",
        "/home/madhekar/work/home-media-app/data/input-data/error/video/missing-data",
        "/home/madhekar/work/home-media-app/data/input-data/error/video/quality",
        "/home/madhekar/work/home-media-app/data/input-data/error/txt/duplicate",
        "/home/madhekar/work/home-media-app/data/input-data/error/txt/missing-data",
        "/home/madhekar/work/home-media-app/data/input-data/error/txt/quality",
        "/home/madhekar/work/home-media-app/data/input-data/error/audio/duplicate",
        "/home/madhekar/work/home-media-app/data/input-data/error/audio/missing-data",
        "/home/madhekar/work/home-media-app/data/input-data/error/audio/quality",
        "/home/madhekar/work/home-media-app/data/final-data/img",
        "/home/madhekar/work/home-media-app/data/final-data/video",
        "/home/madhekar/work/home-media-app/data/final-data/txt",
        "/home/madhekar/work/home-media-app/data/final-data/audio",
    ]

    print(f"{'Folder':<30} | {'File Count':<15} | {'Size (GB)':<15}")
    print("-" * 64)

    src_list, rlist = ['madhekar', 'Samsung USB'], []
    prefix = "/home/madhekar/work/home-media-app/data/"
    for src in src_list:
        for folder in folders_to_check:
            folder = os.path.join(folder, src)
            if os.path.isdir(folder):
                count, size_bytes = get_folder_metrics(folder)
                size_gb = bytes_to_gb(size_bytes)
                ftrim = folder.removeprefix(prefix)
                ftrim = ftrim.replace("/error","")
                # print(ftrim)
                npath = os.path.normpath(ftrim)
                path_list = npath.split(os.sep)
                if len(path_list) ==3:
                    path_list[2] = "data"
                print(path_list)
                # print(f"{ftrim:<30} | {count:<15} | {size_gb:<15.4f}")
                rlist.append({"source": src, "data_stage": path_list[0], "data_type": path_list[1], "data_attrib": path_list[2],  "count": count, "size": size_gb})

            else:
                ftrim = folder.removeprefix(prefix)
                ftrim = ftrim.replace("/error", "")
                # print(ftrim)
                npath = os.path.normpath(ftrim)
                path_list = npath.split(os.sep)
                if len(path_list) == 3:
                    path_list[2] = "data"
                print(path_list)
                # print(f"{ftrim:<30} | {0:<15} | {0:<15.4f} ")
                rlist.append({"source": src, "data_stage": path_list[0], "data_type": path_list[1], "data_attrib": path_list[2],  "count": 0, "size": 0.0})
                pass

    df = pd.DataFrame(rlist, columns=["source", "data_stage", "data_type", "data_attrib", "count", "size"])
    print(df)

    values_to_delete = ['duplicate','missing-data','quality']
    dft = df[~((df['data_stage'] == "final-data") & (df['data_attrib'].isin(values_to_delete)))]
    print(dft)

    df_input = dft[~(dft['data_stage'] == "final-data")]
    df_final = dft[~(dft['data_stage'] == "input-data")]
    # out = dft.pivot_table(index=["source", "data_stage", "data_type"], columns=["data_attrib"], values=["count", "size"])
    # print(out)
    #dft = dft[~(dft['source'] == "Samsung USB")]
    return df_input, df_final

def multi_level_pie(dfs):

    dfs["legend_label"] = dfs["data_stage"] + "::" + dfs["data_type"] + dfs["data_attrib"] + '('+ str(dfs["count"]) +')'
    # Filter data for each level and create separate charts
    # Level 1 (Center) - The total value would ideally be a single point or a calculation
    # This example uses placeholder values for level 1 to show the structure
    chart1 = (
        alt.Chart(dfs[dfs["data_type"] == "img"])
        .mark_arc(outerRadius=25, innerRadius=5)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("data_attrib:N", legend=None),
            order=alt.Order("data_stage:N", sort="descending"),
            tooltip=["source", "data_stage", "data_type", "data_attrib", "count"],
        )
    )

    # Level 2 (Middle Ring)
    chart2 = (
        alt.Chart(dfs[dfs["data_type"] == "video"])
        .mark_arc(outerRadius=45, innerRadius=25)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("legent_label:N", legend=alt.Legend(orient="right", title="Cateogry (Value)")),
            order=alt.Order("data_stage:N", sort="descending"),
            tooltip=["source", "data_stage","data_attrib", "data_type", "count"],
        )
    )

    # Level 3 (Outer Ring) - this data would need to sum up correctly to the parent level for a true sunburst
    # This example uses the 'parent' to group colors but 'category' to define slices
    chart3 = (
        alt.Chart(dfs[dfs["data_type"] == "txt"])
        .mark_arc(outerRadius=65, innerRadius=45)
        .encode(
            theta=alt.Theta("count:Q"),
            # Use parent for general color scheme, category for individual slices
            color=alt.Color("legent_label:N"),  # , scale=alt.Scale(range="source")),
            order=alt.Order("source:N"),
            tooltip=["source", "data_stage", "data_type", "data_attrib", "count"],
        )
    )

    # Layer the charts using the '+' operator
    # The charts are overlaid on the same set of axes
    multilevel_pie_chart = (chart1 + chart2 + chart3).properties(
        title="Multilevel Pie Chart (Sunburst) Example"
    )

    # Display the chart
    st.altair_chart(multilevel_pie_chart)


def show_hirarchy(df):
    c = alt.Chart(df).mark_bar().encode(
        x=alt.X('count:Q', scale=alt.Scale(zero=False)),
        y=alt.Y('size:Q', title='dick usage GB'),
        color='data_type:N'
    ).properties(
        title="disc usage"
    )
    c_=alt.Chart(df).mark_bar().encode(
        x='count',
        y=alt.Y('size:Q'),
        color='data_stage:N'
    )

    st.altair_chart(c | c_)

def filter_selection(df):
    # 1. Define the first dropdown selection
    source_selection = alt.selection_point(
        fields=["source"], bind="legend", name="source"
    )
    # For a dropdown *widget*, use bind=alt.binding_select(options=...)
    source_dropdown = alt.binding_select(
        options=sorted(df["source"].unique().tolist()), name="Select source "
    )
    source_selection = alt.selection_point(fields=["source"], bind=source_dropdown)

    # # 2. Define the second dropdown selection
    # data_stage_dropdown = alt.binding_select(
    #     options=sorted(df["data_stage"].unique().tolist()), name="Select data stage "
    # )
    # data_stage_selection = alt.selection_point(
    #     fields=["data_stage"], bind=data_stage_dropdown
    # )

    #3. Define the third selection
    data_type_dropdown = alt.binding_select(
        options=sorted(df["data_type"].unique().tolist()), name="Select data type "
    )
    data_type_selection = alt.selection_point(
        fields=["data_type"], bind=data_type_dropdown
    )

    # Create the chart
    chart = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("count:Q", axis=alt.Axis(grid=True, gridColor='grey')),
            y=alt.Y("size:Q", axis=alt.Axis(grid=True, gridColor="grey")),
            size="source:N",
            color="data_attrib:N",
            shape="data_type:N",
            tooltip=["source", "data_stage", "data_type", "data_attrib", 'count', 'size'],
        )
        .add_params(source_selection,  data_type_selection)
        .transform_filter(
            # Combine both selections using logical AND
            source_selection & data_type_selection
        )
    ).interactive()
    # st.markdown("""<style> 
    #             .vega-bind {
    #             text-align:right;
    #             }</style> """, unsafe_allow_html=True)
    st.altair_chart(chart)

filter_selection(acquire_data()[0])    