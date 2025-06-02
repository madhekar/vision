import orjson
import ijson  # You may need ijson to help parse the JSON stream

# def process_json_in_batches(file_path, batch_size):
#     """
#     Generator function to process JSON data from a file in batches.

#     Args:
#         file_path (str): The path to the JSON file.
#         batch_size (int): The number of items per batch.

#     Yields:
#         list: A batch of processed JSON data.
#     """
#     batch = []
#     # Using ijson to parse JSON incrementally
#     with open(file_path, 'rb') as file:
#         parser = ijson.items(file, 'item')  # Adjust 'item' based on your JSON structure
#         for item in parser:
#             batch.append(item)
#             if len(batch) >= batch_size:
#                 yield batch
#                 batch = []
#         # Yield the remaining items in the last batch
#         if batch:
#             yield batch

# Example of processing within the generator
def process_json_in_batches(file_path, batch_size):
    batch = []
    with open(file_path, 'rb') as file:
        parser = ijson.items(file, 'item')
        for item in parser:
            # Process the item with orjson
            processed_item = orjson.dumps(item)  # Example: serialize with orjson
            batch.append(processed_item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# Example of iterating and consuming the batches
for batch in process_json_in_batches('image_people_names_emotions.json', 10):
    # Process each batch
    print(f"Processing batch of size: {len(batch)}")
    print(batch)
    # Further processing of the batch...