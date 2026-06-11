import shutil
import os

# Read mounted filesystems from the system
with open('/proc/mounts', 'r') as f:
    mounts = f.readlines()

print(f"{'Partition':<15} {'Size':<15}")
print("-" * 30)

for line in mounts:
    parts = line.split()
    device = parts[0]
    mount_point = parts[1]

    # Filter out pseudo-filesystems (like sysfs, tmpfs, proc)
    if device.startswith('/dev/'):
        try:
            usage = shutil.disk_usage(mount_point)
            total_size_gb = usage.total / (1024 ** 3)
            print(f"{device:<15} {total_size_gb:.2f} GB")
        except Exception:
            pass