import fs

# Open the filesystem using a specific URL format
# The format is 'smb://username:password@server/share'
try:
    smb_fs = fs.open_fs('smb://madhekar:Manya1@us@192.168.68.121/home-media-app')

    # Now you can use standard PyFilesystem methods
    if smb_fs.exists("remote_file.txt"):
        with smb_fs.open("remote_file.txt", "r") as remote_file:
            content = remote_file.read()
            print(content)

    # Example of writing a file
    smb_fs.writetext("new_file.txt", "Hello, world!")

except Exception as e:
    print(f"Error accessing SMB share: {e}")
