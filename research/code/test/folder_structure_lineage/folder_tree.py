import os

def display_tree(path, indent=''):
    """Displays the directory tree structure."""

    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            print(indent + '├── ' + entry)
            display_tree(full_path, indent + '|   ')
        else:
            print(indent + '└── ' + entry)

if __name__ == "__main__":
    path = '.'  # Replace with the desired directory path
    display_tree('/home/madhekar/work/home-media-app/data/raw-data/AnjaliBackup')