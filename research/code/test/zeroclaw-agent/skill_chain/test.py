import subprocess
import sys

arg1 = sys.argv[1]
arg2 = sys.argv[2]
try:
    result = subprocess.run(
        [sys.executable, "script.py", arg1, arg2],  # Command and arguments
        capture_output=True,                           # Saves stdout and stderr internally
        text=True,                                     # Automatically decodes bytes to strings
        check=True                                     # Raises an exception if exit code != 0
    )
    
    # Access the captured text
    print("Script output:", result.stdout)

except subprocess.CalledProcessError as e:
    print(f"Script failed with exit code {e.returncode}")
    print("Error message:", e.stderr)
