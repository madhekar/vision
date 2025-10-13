import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from multiprocessing import Process, Queue

def worker_predict(model, input_queue, output_queue):
    while True:
        item = input_queue.get()
        if item is None:  # Sentinel to stop the worker
            break
        image_data, original_index = item
        predictions = model.predict(image_data)
        output_queue.put((predictions, original_index))

def run_multiprocess_inference(image_paths, num_workers=4):
    model = MobileNetV2(weights='imagenet')
    input_queue = Queue()
    output_queue = Queue()
    workers = []

    for i in range(num_workers):
        p = Process(target=worker_predict, args=(model, input_queue, output_queue))
        p.start()
        workers.append(p)

    # Put preprocessed images into the input queue
    for i, path in enumerate(image_paths):
        preprocessed_img = preprocess_image_fn(path)
        input_queue.put((preprocessed_img, i))

    # Add sentinels to stop workers
    for _ in range(num_workers):
        input_queue.put(None)

    # Collect results
    results = [None] * len(image_paths)
    for _ in range(len(image_paths)):
        predictions, original_index = output_queue.get()
        results[original_index] = predictions

    for p in workers:
        p.join()

    return results

# Example usage:
# image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
# all_predictions = run_multiprocess_inference(image_paths)
