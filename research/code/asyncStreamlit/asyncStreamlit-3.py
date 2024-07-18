import re

import streamlit as st


# Simple persistent state: The dictionary returned by `get_state()` will be
# persistent across browser sessions.
@st.cache_data()
def get_state():
    return {}


# The actual creation of the widgets is done in this function.
# Whenever the selection changes, this function is also used to refresh the input
# widgets so that they reflect their new state in the browser when the script is re-run
# to get visual updates.
def display_widgets():
    users = [(1, "Jim"), (2, "Jim"), (3, "Jane")]
    users.sort(key=lambda user: user[1])  # sort by name
    options = ["%s (%d)" % (name, id) for id, name in users]
    index = [i for i, user in enumerate(users) if user[0] == state["selection"]][0]

    return (
        number_placeholder.number_input(
            "ID",
            value=state["selection"],
            min_value=1,
            max_value=3,
        ),
        option_placeholder.selectbox("Name", options, index),
    )


state = get_state()

# Set to the default selection
if "selection" not in state:
    state["selection"] = 1

# Initial layout
number_placeholder = st.sidebar.empty()
option_placeholder = st.sidebar.empty()

# Grab input and detect changes
selected_number, selected_option = display_widgets()

input_changed = False

if selected_number != state["selection"] and not input_changed:
    # Number changed
    state["selection"] = selected_number
    input_changed = True
    display_widgets()

selected_option_id = int(re.match(r"\w+ \((\d+)\)", selected_option).group(1))
if selected_option_id != state["selection"] and not input_changed:
    # Selectbox changed
    state["selection"] = selected_option_id
    input_changed = True
    display_widgets()

st.write(f"The selected ID was: {state['selection']}")