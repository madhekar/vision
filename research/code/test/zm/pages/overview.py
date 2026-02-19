import streamlit as st
from utils.overview_util import overview as ovr
'''
The "inotify watch limit reached" error (often
Error: ENOSPC or System limit for number of file watchers reached) occurs on Linux when applications like VS Code, Dropbox, or Node.js exceed the maximum number of files/directories allowed to be monitored. The default limit is typically 8192. To fix this, increase the fs.inotify.max_user_watches sysctl parameter. 
Immediate Fix (Temporary until reboot):
Run this command to increase the limit to a higher value, such as 524,288:
sudo sysctl fs.inotify.max_user_watches=524288
sudo sysctl -p 
Permanent Fix:

    Append the limit to sysctl.conf:
    echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
    Apply the changes:
    sudo sysctl -p 

Verification:
Check the current limit by running:

cat /proc/sys/fs/inotify/max_user_watches
 
Common Causes:

    Large Projects: Watching deep file trees (e.g., thousands of files in node_modules or .git).
    Multiple Tools: Running multiple IDEs, file watchers, or build tools.
    Docker/Containers: Running many containers, each with its own watchers. 

If the issue persists, you may need to also increase max_user_instances
'''
st.header("OVERVIEW: DISK STORAGE", divider="gray")
with st.spinner('In Progress...'):
   ovr.execute()

