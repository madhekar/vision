import psutil

# Get all mounted disk partitions
partitions = psutil.disk_partitions()

print(f"{'Device':<15} {'Mountpoint':<20} {'Fstype':<10} {'Total Size':<15}")
print("-" * 65)

for partition in partitions:
    try:
        # Get usage statistics for the partition's mount point
        usage = psutil.disk_usage(partition.mountpoint)
        total_size_gb = usage.total / (1024 ** 3)  # Convert bytes to GB

        print(f"{partition.device:<15} {partition.mountpoint:<20} {partition.fstype:<10} {total_size_gb:.2f} GB")
    except PermissionError:
        print(f"{partition.device:<15} {partition.mountpoint:<20} {partition.fstype:<10} Permission Denied")
