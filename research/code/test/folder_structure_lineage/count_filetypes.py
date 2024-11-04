import os
import collections

def count_file_types(directory):
    file_counts = collections.defaultdict(int)
    for root, _, files in os.walk(directory, topdown=True):
        for file in files:
            _, ext = os.path.splitext(file)
            file_counts[ext] += 1
    return file_counts

directory_path = '../../../'
file_counts = count_file_types(directory_path)

for ext, count in file_counts.items():
    print(f'{ext}: {count}')