import os
'''
//[server-ip]/[share-path] /mnt/myshare cifs credentials=/etc/samba/myshare.cred,iocharset=utf8,nounix,file_mode=0777,dir_mode=0777 0 0
'''

mount_point = "/Users/Shared/zmdata" # Use an existing or newly created local path
# The command might require root/sudo privileges
os.system(f"mount_smbfs //madhekar:Manya1@us@192.168.68.121/home-media-app {mount_point}")
