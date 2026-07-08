import subprocess

# Run the lsblk command to get detailed partition info
result = subprocess.run(['lsblk', '-o', 'NAME,SIZE,FSTYPE,MOUNTPOINT'], capture_output=True, text=True)

print("Disk Layout:")
print(result.stdout)