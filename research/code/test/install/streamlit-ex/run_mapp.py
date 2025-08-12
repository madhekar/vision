# run_app.py
import streamlit.web.cli as stcli
import sys
import os
'''
sudo find /usr -type f -size +10M -printf '%s %p\n' | sort -nrk 1 | numfmt --to=iec --field=1 | head -n 40
'''
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