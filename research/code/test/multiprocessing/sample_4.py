from concurrent.futures import ProcessPoolExecutor
import time

def process_chunk(data_chunk):
    # Simulate some processing on the data chunk
    time.sleep(0.1)
    return sum(data_chunk)

if __name__ == "__main__":
    data = list(range(1000))
    chunk_sizes = [50, 100, 200]
    
    for chunk_size in chunk_sizes:
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = executor.map(process_chunk, [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)])
            total_sum = sum(results)
        end_time = time.time()
        print(f"Chunk size: {chunk_size}, Total sum: {total_sum}, Time taken: {end_time - start_time:.4f} seconds")
