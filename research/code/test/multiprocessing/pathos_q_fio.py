import pathos.multiprocessing as mp
import multiprocessing as m
import queue
import time
import os

def writer(q, filename):
    """Writes data to a file from a queue."""
    with open(filename, 'w') as f:
        while True:
            try:
                data = q.get(timeout=1)  # Wait for 1 second, raise Empty if nothing
                if data is None:
                    break  # Stop if None is received
                f.write(f"{data}\n")
                f.flush()  # Ensure data is written immediately
            except queue.Empty:
                continue

def reader(q, filename):
    """Reads data from a file and puts it into a queue."""
    if not os.path.exists(filename):
      return
    with open(filename, 'r') as f:
        for line in f:
            q.put(line.strip())
    q.put(None)  # Signal the writer to stop

if __name__ == '__main__':
    q = m.Queue()
    filename = './data.txt'
    filename_o = './data_o.txt'

    # Create processes
    writer_p = m.Process(target=writer, args=(q, filename_o))
    reader_p = m.Process(target=reader, args=(q, filename))

    # Start reader first to avoid writer running indefinitely if file is empty
    reader_p.start()
    writer_p.start()
    
    # Simulate data being generated and added to the queue (before reading)
    for i in range(5):
        q.put(f"Data {i}")
        time.sleep(0.5)

    # Wait for processes to complete
    reader_p.join()
    writer_p.join()

    print("Done.")
