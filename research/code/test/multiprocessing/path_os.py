import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool

"""
https://edbennett.github.io/high-performance-python/01-getting-started/index.html
"""
def process_csv(file):
    print(file)
    df = pd.read_csv(file)
    # Perform some data processing
    return df.describe()


csv_files = ['../folder_structure_lineage/locations/default.csv', 
             '../folder_structure_lineage/locations/default.csv', 
             '../folder_structure_lineage/locations/default.csv', 
             '../folder_structure_lineage/locations/default.csv']

# Create a processing pool with 4 workers
pool = Pool(4)

# Process each CSV file in parallel
results = pool.map(process_csv, csv_files)

for result in results:
    print(result)