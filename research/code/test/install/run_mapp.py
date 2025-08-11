# run_app.py
import streamlit.web.cli as stcli
import sys
import os

def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path

if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("mapp.py"),
        "--global.developmentMode=false", # Optional: disable development mode
    ]
    sys.exit(stcli.main())