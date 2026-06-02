import pandas as pd

# Define the batch/chunk size (number of rows per batch)
chunk_size = 10000
chunks = []

# Use chunksize to return an iterator
with pd.read_json("large_file.json", lines=True, chunksize=chunk_size) as reader:
    for chunk in reader:
        # Process your batch chunk here if needed
        chunks.append(chunk)

# Combine chunks into a single DataFrame
final_df = pd.concat(chunks, ignore_index=True)

print(final_df.head())