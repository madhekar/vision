from multiprocessing import Pool, Manager
import tqdm
import time

def my_function(x):
    time.sleep(0.1)
    return x * 2

def update_progress(result, pbar):
    pbar.update(1)

if __name__ == '__main__':
    inputs = range(100)
    with Manager() as manager:
        pbar = tqdm.tqdm(total=len(inputs))
        with Pool(4) as pool:
            async_results = [pool.apply_async(my_function, (x,), callback=lambda res: update_progress(res, pbar)) for x in inputs]
            results = [ar.get() for ar in async_results]
        pbar.close()
    print(results)