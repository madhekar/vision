import pandas as pd
from st_aggrid import AgGrid, GridUpdateMode, DataReturnMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder


df = pd.DataFrame({"image_path":["https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Elon_Musk_Royal_Society.jpg/800px-Elon_Musk_Royal_Society.jpg",
                                    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Bill_Gates_-_Nov._8%2C_2019.jpg/390px-Bill_Gates_-_Nov._8%2C_2019.jpg"],
                    "Name":['Elon', 'Bill']
                    })

options_builder = GridOptionsBuilder.from_dataframe(df)


image_nation = JsCode("""function (params) {
        console.log(params);
        var element = document.createElement("span");
        var imageElement = document.createElement("img");
    
        imageElement.src = params.data.image_path;
        imageElement.width="40";
        imageElement.height="40";

        element.appendChild(imageElement);
        element.appendChild(document.createTextNode(params.value));
        return element;
        }""")
options_builder.configure_column('Name', cellRenderer=image_nation)


grid_options = options_builder.build()

grid_return = AgGrid(df,
                    grid_options,
                    theme="streamlit",
                    allow_unsafe_jscode=True,
                    ) 