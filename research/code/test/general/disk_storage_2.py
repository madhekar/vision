import shutil
import psutil

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