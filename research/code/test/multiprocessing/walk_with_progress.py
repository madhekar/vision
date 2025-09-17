import os
from tqdm import tqdm

def walk_with_progress(root_dir):
    """
    Walks through a directory tree with a tqdm progress bar.
    """
    # First, estimate the total number of items to iterate over for a more accurate progress bar
    # This can be done by a quick pre-scan or by using a dynamic total if exact count is not feasible.
    # For a simple file count:
    total_items = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        total_items += len(dirnames) + len(filenames)

    with tqdm(total=total_items, desc=f"Processing {root_dir}") as pbar:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Process directories and files here
            # For demonstration, we just update the progress bar for each item
            for _ in dirnames:
                pbar.update(1)
            for _ in filenames:
                pbar.update(1)
            # You can also perform actual operations on files/directories here
            # For example:
            # for filename in filenames:
            #     filepath = os.path.join(dirpath, filename)
            #     # Perform operations on filepath
            #     pbar.update(1) # Update after processing each file

walk_with_progress("/home/madhekar/work/home-media-app/data/input-data/img/madhekar")            