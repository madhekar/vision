import pathos.multiprocessing as mp
import time
from itertools import islice

# Hypothetical LLM inference function
def llm_inference(text):
    time.sleep(0.1)  # Simulate LLM processing time
    return f"LLM processed: {text}"

def getChunks(c_size =10):
    t_list = [f"Text {i}" for i in range(2000)]  # Example list of text
    for i in range(0, len(t_list), 10):
        yield t_list[i:i+10] 

if __name__ == '__main__':

    # Sequential processing
    start_time = time.time()
    sequential_results = [llm_inference(text) for text in getChunks(10)]
    sequential_time = time.time() - start_time

    # Parallel processing using pathos
    pool = mp.ProcessPool(4)  # Create a pool with 4 processes
    start_time = time.time()
    parallel_results = pool.map(llm_inference, getChunks(10))
    parallel_time = time.time() - start_time
    pool.close()
    pool.join()

    print("Sequential processing results:")
    for result in sequential_results:
        print(result)
    print(f"Sequential time: {sequential_time:.2f} seconds\n")

    print("Parallel processing results:")
    for result in parallel_results:
        print(result)
    print(f"Parallel time: {parallel_time:.2f} seconds")

    print(f"\nSpeedup: {sequential_time/parallel_time:.2f}x")