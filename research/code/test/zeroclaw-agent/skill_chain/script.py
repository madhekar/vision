import sys

# Check if both arguments are provided
if len(sys.argv) < 3:
    print("Error: Please provide two arguments.")
    print("Usage: python sys_example.py <arg1> <arg2>")
    sys.exit(1)

# Read the arguments from the command line list
argument_one = sys.argv[1]
argument_two = sys.argv[2]

# Use the arguments
print(f"First argument received: {argument_one}")
print(f"Second argument received: {argument_two}")