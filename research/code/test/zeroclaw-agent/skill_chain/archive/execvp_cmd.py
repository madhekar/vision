import os

# Define the command to look up in PATH
command = "ls"

# Define the argument list. 
# REMEMBER: The first element MUST be the program name!
arguments = ["ls", "-la"]

print("This line will print.")

try:
    os.execvp(command, arguments)
except OSError as e:
    print(f"Execution failed: {e}")

# This line will NEVER be reached if execvp succeeds
print("This line will NOT print.")
