import multiprocessing as mp
import functools as ft
from stqdm import stqdm

def worker(file_path, session_target_path, queue_obj):
    result = func(target=file_path)
    queue_obj.put(result, timeout=60)

def listener(queue_obj, n_files):
    model = load_model() # tf model
    with stqdm(desc='text', total=n_files) as progress:
        while True:
            try:
                message = queue_obj.get(timeout=20)

            except Empty:
                break

            score = make_inference()
            progress.update()

            print(score)

def multi_process_directory(file_paths):
    manager = mp.Manager()
    queue = manager.Queue()    
    pool = mp.Pool(mp.cpu_count() - 2)
    
    # Now using a single listener
    pool.apply_async(listener, (queue, len(file_paths)))

    func = ft.partial(worker, session_target_path, queue)
    for _ in stqdm(pool.imap(func, file_paths), total=len(file_paths)):
        pass

    pool.close()
    pool.join()