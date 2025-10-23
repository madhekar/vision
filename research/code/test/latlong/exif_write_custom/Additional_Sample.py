import subprocess
import json

class ExifTool(object):
    """
    A context manager for running exiftool in persistent mode.
    """
    def __init__(self, executable='/usr/bin/exiftool'):
        self.executable = executable
        self.process = None

    def __enter__(self):
        self.process = subprocess.Popen(
            [self.executable, "-stay_open", "True", "-@", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Use text mode for easier string handling
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            self.process.stdin.write("-stay_open\nFalse\n")
            self.process.stdin.flush()
            self.process.communicate()
            self.process = None

    def execute(self, *args):
        """
        Executes a single exiftool command and returns the output.
        """
        if not self.process:
            raise RuntimeError("ExifTool process is not running.")
            
        cmd = [str(arg) for arg in args] + ["\n"]
        print(f"->> {cmd}")
        self.process.stdin.write("\n".join(cmd))
        self.process.stdin.flush()
        
        output = ""
        while True:
            line = self.process.stdout.readline()
            print(f"-line-> {line}")
            if line.strip() == "{ready}":
                break
            output += line
        
        # Read and discard any stderr output for cleanup
        self.process.stderr.readline() 
        
        return output.strip()

    def write_metadata(self, filepath, metadata):
        """
        Writes custom metadata to a file.
        
        Args:
            filepath (str): The path to the file to modify.
            metadata (dict): A dictionary of tags and values.
        """
        command = []
        for tag, value in metadata.items():
            print(f"--> {tag} -> {value}")
            command.append(f"-{tag}={value}")
        command.append(filepath)
        
        # Add the "-overwrite_original" flag to prevent creating backup files
        # Alternatively, omit this flag to create "_original" backups
        command.append("-overwrite_original")

        output = self.execute(*command)
        print(f"--> {output}")
        return output

### Example usage

#To write a custom tag, such as `XMP-dc:Source`, to a JPEG file:

#python
#from your_module import ExifTool # Assuming the class is in a file called your_module.py

# Create a dummy image file for demonstration
with open("/home/madhekar/temp/faces/Bhiman/bhiman3.png", "w") as f:
    f.write("This is not a real JPEG file, but exiftool will add metadata.")

# Define the custom metadata to write
custom_metadata = {
    "XMP-dc:Source": "My custom source value",
    "XMP-iptcCore:CreatorContactInfo": "example@example.com"
}

try:
    with ExifTool() as et:
        # Use the context manager to ensure the exiftool process is handled correctly
        print("Writing metadata to test_image.jpg...")
        output = et.write_metadata("test_image.jpg", custom_metadata)
        print(output)
        
        # Verify the change by reading the metadata
        print("\nReading metadata to confirm:")
        read_output = et.execute("-s3", "-XMP-dc:Source", "test_image.jpg")
        print(read_output)

except FileNotFoundError:
    print("Error: exiftool executable not found. Make sure it's installed and in your PATH.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Cleanup the dummy file
    import os
    if os.path.exists("test_image.jpg_original"):
        os.remove("test_image.jpg_original")
    if os.path.exists("test_image.jpg"):
        os.remove("test_image.jpg")
