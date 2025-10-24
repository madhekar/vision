import subprocess
import shlex
import pandas as pd

"""
    Writes user comments from a DataFrame to image files using exiftool in batch mode.
    Args: df (pd.DataFrame): DataFrame with 'filepath' and 'comment' columns.
    """
def batch_write_comments(ld):

    command = ['exiftool', '-stay_open', 'True', '-@', '-']
    
    try:
        # Start the persistent ExifTool process
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Use text mode for standard streams
        )

        for row in ld:
            filepath = row['filepath']
            comment = row['comment']
            
            # Construct the arguments for each file
            # Use '-comment' to write to the standard UserComment tag
            args = [f'-UserComment={comment}', '-overwrite_original', filepath]
            
            # Send arguments to the ExifTool process's stdin, followed by '-execute'
            arg_string = '\n'.join(shlex.quote(arg) for arg in args) + '\n-execute\n'
            proc.stdin.write(arg_string)
            proc.stdin.flush()
            
            # Wait for the output from ExifTool and check for errors
            output_line = proc.stdout.readline()
            if "error" in output_line.lower():
                print(f"Error writing to {filepath}: {output_line}")
            else:
                print(f"Successfully wrote comment to {filepath}")

        # Close the persistent exiftool process
        proc.stdin.write('-stay_open\nFalse\n')
        proc.stdin.flush()
        proc.wait(timeout=5)

    except FileNotFoundError:
        print("Error: exiftool not found. Make sure it's installed and in your PATH.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__=='__main__':
    # Call the function with your DataFrame
    dl = [{"filepath":"/home/madhekar/temp/faces/Sachi/saach5.jpg", "comment":"people"},
          {"filepath":"/home/madhekar/temp/faces/Sachi/saachi.png", "comment":"people"} 
          ]
    df = pd.DataFrame(dl)
    batch_write_comments(dl)