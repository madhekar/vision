import shutil
import psutil

import psutil

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


def storage_stats(path):
    # Get disk usage information
    usage = shutil.disk_usage(path)

    # Print disk usage information
    print("Disk Usage Information:")
    print("Total Space:", usage.total / (1024**3), " GB")
    print("Used Space:", usage.used / (1024**3), " GB")
    print("Free Space:", usage.free / (1024**3), " GB")

    # Convert to GB for readability
    print("\nIn Gigabytes:")
    print("Total Space: {:.2f} GB".format(usage.total / (1024**3)))


if __name__ == "__main__":
    get_partitions_info()

    # Get disk usage information
    path = '.'
    usage = psutil.disk_usage(path)

    # Print disk usage information
    print("Disk Usage Information:")
    print("Total Space:", usage.total)
    print("Used Space:", usage.used)
    print("Free Space:", usage.free)

    # Get additional disk partition information
    partitions = psutil.disk_partitions()
    for partition in partitions:
        print(f"\nPartition: {partition.device}")
        print(f"File System: {partition.fstype}")
        print(f"Mount Point: {partition.mountpoint}")

    storage_stats('/')    