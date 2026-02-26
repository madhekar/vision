import os
mount_point = "/Users/Shared/zmdata" # Use an existing or newly created local path
# The command might require root/sudo privileges
os.system(f"mount_smbfs //madhekar:Manya1@us@madhekar-um690/home-media-app {mount_point}")
