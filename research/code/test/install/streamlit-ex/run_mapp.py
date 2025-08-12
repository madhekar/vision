# run_app.py
import streamlit.web.cli as stcli
import sys
import os
from pathlib import Path
'''
sudo find /usr -type f -size +10M -printf '%s %p\n' | sort -nrk 1 | numfmt --to=iec --field=1 | head -n 40
'''
def resolve_path(path) -> str:
    base_path = getattr(sys, "_MEIPASS", os.getcwd())
    #resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return str(Path(base_path) / path)

if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("mapp.py"),
        "--global.developmentMode=false",
         "--server.headless=true" # Optional: disable development mode
    ]
    sys.exit(stcli.main())