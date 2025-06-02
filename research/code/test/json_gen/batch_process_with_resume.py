import ijson
import orjson
import os

class process_json:
    def __init__(self, _json_file, _batch_size, _progress_file):
        self.json_file = _json_file
        self.batch_size = _batch_size
        self.progress_file = _progress_file

    def process_large_json(self, resume_index=0):
        processed_count = resume_index

        # Use ijson to stream JSON objects from the file
        with open(self.json_file, 'rb') as fp:
            # Assuming the JSON is a list of objects 
            batch = []
            for index, item in enumerate(ijson.items(fp, 'item'), start=0):
                if index < resume_index:
                    continue  # Skip items already processed

                # Process the item (e.g., transform, enrich)
                processed_item = self.process_item(item)

                # Store the processed item or yield it in batches
                # (In this example, we'll process individually)
                # You might append to a batch list here and yield when batch_size is reached
                processed_count = index + 1
                # Save the current index to a file for resuming
                self.save_progress(processed_count)

                batch.append(processed_item)
                if len(batch) >= self.batch_size:
                    self.process_batch(batch)
                    batch = []
            if batch:
                self.process_batch(batch) 

    def process_batch(self, batch):
        print(f'begin -\n {batch} \n- end.')


    def process_item(self, item):
        # Use orjson.dumps() for efficient serialization 
        return orjson.dumps(item)

    def save_progress(self, index):
        with open(self.progress_file, "w") as f:
            f.write(str(index))

    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as f:
                return int(f.read())
        return 0

# Usage
file_path = "image_people_names_emotions.json" 
progress_file = 'json_process_progress.txt'
pj = process_json(file_path, 10, progress_file)
resume_index = pj.load_progress()
pj.process_large_json(resume_index)