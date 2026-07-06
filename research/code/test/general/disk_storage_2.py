import shutil

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


storage_stats('/')    