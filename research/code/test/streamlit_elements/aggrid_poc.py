from st_aggrid import JsCode, AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Paris', 'Tokyo']
}

df = pd.DataFrame(data)

cellStyle = JsCode(
    r"""
    function(cellClassParams) {
         if (cellClassParams.data.gold > 3) {
            return {'background-color': 'gold'}
         }
         return {};
        }
   """)

grid_builder = GridOptionsBuilder.from_dataframe(df)
grid_options = grid_builder.build()

grid_options['defaultColDef']['cellStyle'] = cellStyle

AgGrid(data=df, gridOptions=grid_options, allow_unsafe_jscode=True, key='grid1')