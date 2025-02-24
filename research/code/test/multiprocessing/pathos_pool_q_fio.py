import multiprocessing as m
import queue
import threading
import time

def process_data(data_queue, output_queue):
    while True:
        try:
            data = data_queue.get(timeout=0.1)
            result = data.strip().upper()  # Simulate processing
            output_queue.put(result)
            data_queue.task_done()
        except queue.Empty:
            break

def write_output(filename, output_queue):
    with open(filename, 'w') as f:
        while True:
            try:
                result = output_queue.get(timeout=0.1)
                f.write(result + '\n')
                output_queue.task_done()
            except queue.Empty:
                break

if __name__ == '__main__':
    input_filename = 'data.txt'
    output_filename = 'data_o.txt'
    num_processes = 4

    data_queue = queue.Queue()
    output_queue = queue.Queue()

    with open(input_filename, 'r') as f:
        for line in f:
            data_queue.put(line)

    pool = m.Pool(10)
    
    for _ in range(num_processes):
      pool.apply_async(process_data, (data_queue, output_queue))
    
    pool.close()
    data_queue.join()
    pool.join()
    
    write_output(output_filename, output_queue)
    output_queue.join()

    print(f"Processing complete. Results written to {output_filename}")