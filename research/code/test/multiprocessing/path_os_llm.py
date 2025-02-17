import pathos.multiprocessing as mp
import time

# Hypothetical LLM inference function
def llm_inference(text):
    time.sleep(0.1)  # Simulate LLM processing time
    return f"LLM processed: {text}"

if __name__ == '__main__':
    texts = [f"Text {i}" for i in range(20)]  # Example list of texts

    # Sequential processing
    start_time = time.time()
    sequential_results = [llm_inference(text) for text in texts]
    sequential_time = time.time() - start_time

    # Parallel processing using pathos
    pool = mp.ProcessPool(4)  # Create a pool with 4 processes
    start_time = time.time()
    parallel_results = pool.map(llm_inference, texts)
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