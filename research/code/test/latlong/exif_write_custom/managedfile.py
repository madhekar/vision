class ManagedFile:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        # Acquire the resource (open the file)
        self.file = open(self.filename, self.mode)
        print(f"File '{self.filename}' opened.")
        return self.file  # Return the opened file object

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Release the resource (close the file)
        if self.file:
            self.file.close()
            print(f"File '{self.filename}' closed.")
        # Handle exceptions if needed (return True to suppress, False to propagate)
        if exc_type:
            print(f"An exception occurred: {exc_val} -> {exc_tb}")
        return False # Propagate exceptions

# Using the context manager
with ManagedFile("my_data.txt", "w") as f:
    f.write("Hello, world!\n")
    # An error could occur here, but __exit__ would still close the file
    # raise ValueError("Something went wrong!") 

print("Outside the with block.")
