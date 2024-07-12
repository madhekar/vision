import asyncio
import pandas as pd
import plotly.express as px

import streamlit as st
from datetime import datetime


CHOICES = [1, 2, 3]


def main():
    print("\nmain...")

    # layout your app beforehand, with st.empty
    # for the widgets that the async function would populate
    graph = st.empty()
    radio = st.radio("Choose", CHOICES, horizontal=True)
    table = st.empty()

    try:
        # async run the draw function, sending in all the
        # widgets it needs to use/populate
        asyncio.run(draw_async(radio, graph, table))
    except Exception as e:
        print(f"error...{type(e)}")
        raise
    finally:
        # some additional code to handle user clicking stop
        print("finally")
        # this doesn't actually get called, I think :(
        table.write("User clicked stop!")


async def draw_async(choice, graph, table):
    # must send in all the streamlit widgets that
    # this fn would interact with...

    # this could possibly work, but layout is tricky
    # choice2 = st.radio('Choose 2', CHOICES)

    while True:
        # this would not work because you'd be creating duplicated
        # radio widgets
        # choice3 = st.radio('Choose 3', CHOICES)

        timestamp = datetime.now()
        sec = timestamp.second

        graph_df = pd.DataFrame(
            {
                "x": [0, 1, 2],
                "y": [max(CHOICES), choice, choice * sec / 60.0],
                "color": ["max", "current", "ticking"],
            }
        )

        df = pd.DataFrame(
            {
                "choice": CHOICES,
                "current_choice": len(CHOICES) * [choice],
                "time": len(CHOICES) * [timestamp],
            }
        )

        graph.plotly_chart(px.bar(graph_df, x="x", y="y", color="color"))
        table.dataframe(df)

        _ = await asyncio.sleep(1)


if __name__ == "__main__":
    main()
