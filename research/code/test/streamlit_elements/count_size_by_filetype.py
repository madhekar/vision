import os
import pandas as pd
import altair as alt
import streamlit as st

def get_file_data(root_dir):
    """Recursively walks a directory and collects file data by type."""
    file_data = {}

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            try:
                path = os.path.join(root, file)
                # Skip symbolic links to prevent errors or infinite loops
                if not os.path.islink(path):
                    ext = os.path.splitext(file)[1].upper() or "No Extension"
                    size = os.path.getsize(path) # size in bytes

                    if ext not in file_data:
                        file_data[ext] = {'Count': 0, 'Size (MB)': 0}
                    
                    file_data[ext]['Count'] += 1
                    file_data[ext]['Size (MB)'] += size
            except OSError:
                print(f"Error accessing file: {path}")

    # Convert sizes from bytes to Megabytes for better visualization scale
    for ext in file_data:
        file_data[ext]['Size (MB)'] = round(file_data[ext]['Size (MB)'] / (1024 * 1024), 2)
        
    return file_data

def create_chart(data_dict):
    """Creates a layered Altair bar chart for count and size."""
    # Convert dictionary to pandas DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index()
    df.columns = ['File Type', 'Count', 'Size (MB)']

    # Ensure correct data types for Altair
    df['Count'] = df['Count'].astype(int)
    df['Size (MB)'] = df['Size (MB)'].astype(float)
    
    # Sort data for consistent chart order
    df = df.sort_values('Size (MB)', ascending=False)

    base = alt.Chart(df).encode(
        y=alt.Y('File Type:N', sort='-x', title='File Type'), # Sort descending by x-value
        tooltip=['File Type', 'Count', 'Size (MB)']
    ).properties(
        title='File Count and Size by Type',
        width=600
    )

    # Bar chart for Size (MB)
    size_chart = base.mark_bar(color='skyblue', opacity=0.7).encode(
        x=alt.X('Size (MB):Q', title='Total Size (MB)'),
    )

    # Text labels for Count on the bars
    text_count = size_chart.mark_text(
        align='left',
        baseline='middle',
        dx=3 # Nudges text to the right of the bar
    ).encode(
        x='Size (MB):Q',
        text='Count:Q',
        color=alt.value('black')
    )

    # Combine the bar chart and text labels
    chart = size_chart + text_count
    return chart

# --- Main execution ---
if __name__ == "__main__":
    # Specify the directory path you want to scan (e.g., current directory '.')
    directory_to_scan = '.' 
    
    file_stats = get_file_data(directory_to_scan)
    
    if file_stats:
        chart = create_chart(file_stats)
        st.altair_chart(chart)
        # Display the chart (works in 