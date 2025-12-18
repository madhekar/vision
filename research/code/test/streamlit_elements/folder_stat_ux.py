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

    # out = dft.pivot_table(index=["source", "data_stage", "data_type"], columns=["data_attrib"], values=["count", "size"])
    # print(out)

    return dft

def multi_level_pie(dfs):
    # Filter data for each level and create separate charts
    # Level 1 (Center) - The total value would ideally be a single point or a calculation
    # This example uses placeholder values for level 1 to show the structure
    chart1 = alt.Chart(dfs[dfs['data_type'] == "img"]).mark_arc(outerRadius=50, innerRadius=0).encode(
        theta=alt.Theta("size:Q"),
        color=alt.Color("data_attrib:N", legend=None),
        order=alt.Order("data_stage:Q", sort="descending"),
        tooltip=['source', 'data_stage', 'data_type', 'size']
    )

    # Level 2 (Middle Ring)
    chart2 = (
        alt.Chart(dfs[dfs["data_type"] == "video"])
        .mark_arc(outerRadius=100, innerRadius=50)
        .encode(
            theta=alt.Theta("size:Q"),
            color=alt.Color("data_attrib:N", legend=None),
            order=alt.Order("data_stage:Q", sort="descending"),
            tooltip=["source", "data_stage", "data_type", "size"],
        )
    )

    # Level 3 (Outer Ring) - this data would need to sum up correctly to the parent level for a true sunburst
    # This example uses the 'parent' to group colors but 'category' to define slices
    chart3 = (
        alt.Chart(dfs[dfs["data_attrib"] == "txt"])
        .mark_arc(outerRadius=150, innerRadius=100)
        .encode(
            theta=alt.Theta("size:Q"),
            # Use parent for general color scheme, category for individual slices
            color=alt.Color("data_stage:N"), #, scale=alt.Scale(range="source")),
            order=alt.Order("source:N"),
            tooltip=["source", "data_stage", "data_type", "size"],
        )
    )

    # Layer the charts using the '+' operator
    # The charts are overlaid on the same set of axes
    multilevel_pie_chart = (chart1 + chart2 + chart3).properties(
        title="Multilevel Pie Chart (Sunburst) Example"
    )

    # Display the chart
    st.altair_chart(multilevel_pie_chart)

multi_level_pie(acquire_data())    