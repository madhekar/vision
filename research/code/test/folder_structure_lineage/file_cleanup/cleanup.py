import os



def check_size(file_path):
    try:
        size = os.path.getsize(file_path)
        print(f"File size: {size} bytes")
        
        # Convert to Megabytes for readability
        print(f"File size: {size / (1024 * 1024):.2f} MB")
    except OSError as e:
        print(f"Error: {e}")


img1 = ""
check_size(img1)