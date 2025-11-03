import streamlit as st



# Define your pages
def home_page():
    st.title("Home Page")
    st.write("Welcome to the home page!")


def data_analysis():
    st.title("About Us")
    st.write("Learn more about our project.")


def settings():
    st.title("Contact Us")
    st.write("Reach out to us for more information.")


# Create a custom navigation menu with st.page_link
st.sidebar.title("Navigation")
st.sidebar.page_link("pages/home_page.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/data_analysis.py", label="About", icon="ℹ️")
st.sidebar.page_link("pages/settings.py", label="Contact", icon="📧")

# Apply custom CSS to the navigation bar (sidebar in this case)
st.markdown(
    """
    <style>
    /* Target the sidebar containing the navigation links */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6; /* Light gray background */
        padding-top: 20px;
    }

    /* Style the navigation links */
    .st-emotion-cache-1cypcdb { /* This class targets the st.page_link elements */
        color: #333333; /* Dark gray text color */
        font-size: 18px;
        padding: 10px 15px;
        margin-bottom: 5px;
        border-radius: 5px;
    }

    /* Style the navigation links on hover */
    .st-emotion-cache-1cypcdb:hover {
        background-color: #e0e0e0; /* Slightly darker gray on hover */
        color: #007bff; /* Blue text color on hover */
    }

    /* Style the active navigation link */
    .st-emotion-cache-1cypcdb[data-selected="true"] { /* Target selected link */
        background-color: #007bff; /* Blue background for active link */
        color: white; /* White text for active link */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render the content of the selected page
# (In a real multipage app, this would be handled by Streamlit's routing)
# For this example, we'll just show a placeholder based on the current file
if st.session_state.get("current_page") == "about_page":
    about_page()
elif st.session_state.get("current_page") == "contact_page":
    contact_page()
else:
    home_page()
