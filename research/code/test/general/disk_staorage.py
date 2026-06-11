import psutil

# # Get all mounted disk partitions
# partitions = psutil.disk_partitions()

# print(f"{'Device':<15} {'Mountpoint':<20} {'Fstype':<10} {'Total Size':<15}")
# print("-" * 65)

# for partition in partitions:
#     try:
#         # Get usage statistics for the partition's mount point
#         usage = psutil.disk_usage(partition.mountpoint)
#         total_size_gb = usage.total / (1024 ** 3)  # Convert bytes to GB

#         print(f"{partition.device:<15} {partition.mountpoint:<20} {partition.fstype:<10} {total_size_gb:.2f} GB")
#     except PermissionError:
#         print(f"{partition.device:<15} {partition.mountpoint:<20} {partition.fstype:<10} Permission Denied")


# Function to convert bytes to human-readable format (GB)
def format_size(size_in_bytes):
    return f"{size_in_bytes / (1024**3):.2f} GB"

def get_partitions_info():
    # Get all active partitions
    partitions = psutil.disk_partitions()
    
    print(f"{'Device':<15} {'Mountpoint':<20} {'Total':<10} {'Filled':<10} {'Available':<10}")
    print("-" * 65)

    for partition in partitions:
        # Ignore CD-ROMs and partition types without a mount point
        if not partition.mountpoint or 'loop' in partition.device:
            continue
            
        try:
            # Fetch disk usage statistics for each partition mount point
            usage = psutil.disk_usage(partition.mountpoint)
            
            total = format_size(usage.total)
            used = format_size(usage.used)
            free = format_size(usage.free)
            
            print(f"{partition.device:<15} {partition.mountpoint:<20} {total:<10} {used:<10} {free:<10}")
        except PermissionError:
            print(f"{partition.device:<15} {partition.mountpoint:<20} {'(Permission Denied)'}")

if __name__ == "__main__":
    get_partitions_info()