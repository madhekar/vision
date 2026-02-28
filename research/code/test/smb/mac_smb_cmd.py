import os
'''
Connect from macOS Client 
-------------------------
Open Finder on your Mac.
Select Go > Connect to Server (or press Command+K).
Enter the address: smb://<linux-mint-ip-address> (e.g.smb://192.168.1.10).
Click Connect.
Select Registered User, enter your Linux Mint username and the password you set in step 3. 

----

cifs-utils
is a collection of essential Linux user-space tools (including mount.cifs) used to mount and manage SMB/CIFS shares, enabling seamless access to Windows file shares, NAS devices, or Azure storage. It works alongside the kernel cifs.ko module to treat network shares as local filesystems. 
Key Features and Commands

    mount.cifs: The primary command used to attach a network resource to a local directory.
    Protocol Support: Supports CIFS, SMB2, and SMB3, allowing secure, modern connections.
    Integration: Facilitates mounting shares on demand or permanently via /etc/fstab.
    Credentials Management: Supports secure password passing via files or keyrings (e.g., pam_cifscreds). 

Installation

    Ubuntu/Debian: sudo apt install cifs-utils
    Fedora/RHEL: sudo dnf install cifs-utils 

Usage Examples
Manual Mount:
bash

sudo mount -t cifs //server/share /mnt/mountpoint -o username=user,password=pass

Temporary Mount with Permissions:
bash

sudo mount -t cifs //nas-server/share /mnt/local -o username=user,iocharset=utf8,file_mode=0777,dir_mode=0777 [12]

Essential Components

    mount.cifs (mounts a CIFS/SMB3 filesystem)
    smbinfo (queries information about CIFS mounts)
    cifscreds (manages user-space CIFS credentials) 

//[server-ip]/[share-path] /mnt/myshare cifs credentials=/etc/samba/myshare.cred,iocharset=utf8,nounix,file_mode=0777,dir_mode=0777 0 0
'''

mount_point = "/Users/Shared/zmdata" # Use an existing or newly created local path
# The command might require root/sudo privileges
os.system(f"mount_smbfs //madhekar:Manya1@us@192.168.68.121/home-media-app {mount_point}")
