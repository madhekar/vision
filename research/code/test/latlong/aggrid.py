import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

def main():
    st.title("AG Grid CRUD Example")

    # Sample data
    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame({
            "Name": ["John", "Mary", "Peter"],
            "Age": [25, 30, 35]
        })

    # Create a grid
    gb = GridOptionsBuilder.from_dataframe(st.session_state.data)
    gb.configure_grid_options(enableRangeSelection=True)
    gridOptions = gb.build()

    grid_response = AgGrid(
        st.session_state.data,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        allow_unsafe_jscode=True
    )
    updated_data = grid_response['data']

    # Add new row
    if st.button("Add"):
        new_row = {"Name": "", "Age": ""}
        st.session_state.data = st.session_state.data.append(new_row, ignore_index=True)

    # Update data
    st.session_state.data = updated_data

    # Delete selected rows
    if st.button("Delete"):
        selected_rows = grid_response['selected_rows']
        if selected_rows:
            selected_indices = [st.session_state.data.index.get_loc(row['id']) for row in selected_rows]
            st.session_state.data = st.session_state.data.drop(selected_indices)

if __name__ == "__main__":
    main()