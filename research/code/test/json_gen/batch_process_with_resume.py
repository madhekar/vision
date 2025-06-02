import ijson
import orjson
import os

def process_large_json(file_path, batch_size=10, resume_index=0):
    processed_count = resume_index

    # Use ijson to stream JSON objects from the file
    with open(file_path, 'rb') as f:
        # Assuming the JSON is a list of objects 
        batch = []
        for index, item in enumerate(ijson.items(f, 'item'), start=0):
            if index < resume_index:
                continue  # Skip items already processed

            # Process the item (e.g., transform, enrich)
            processed_item = process_item(item)

            # Store the processed item or yield it in batches
            # (In this example, we'll process individually)
            # You might append to a batch list here and yield when batch_size is reached
           
            batch.append(process_item)
            if len(batch) >= batch_size:
                process_batch(batch)
                batch = []
            # if batch:
            #     process_batch(batch)    

            processed_count = index + 1
            # Save the current index to a file for resuming
            save_progress(processed_count)

            # Example processing (replace with your logic)
            #print(f"Processed item at index {index}: {processed_item}")


def process_batch(batch):
    print(f'begin - {batch} - end.')


def process_item(item):
    # Your processing logic here
    # Use orjson.dumps() for efficient serialization if needed
    return orjson.dumps(item)

def save_progress(index):
    with open("progress.txt", "w") as f:
        f.write(str(index))

def load_progress():
    if os.path.exists("progress.txt"):
        with open("progress.txt", "r") as f:
            return int(f.read())
    return 0

# --- Usage ---
file_path = "image_people_names_emotions.json"  # Replace with your file path
resume_index = load_progress()
process_large_json(file_path, resume_index=resume_index)