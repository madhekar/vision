import time
import random
from tqdm import tqdm
from multiprocessing import Pool, freeze_support, RLock

def func(pid, n):

    tqdm_text = "#" + "{}".format(pid).zfill(3)

    current_sum = 0
    with tqdm(total=n, desc=tqdm_text, position=pid+1) as pbar:
        for i in range(1, n+1):
            current_sum += i
            time.sleep(0.05)
            pbar.update(1)
    
    return current_sum

def main():

    freeze_support() # For Windows support

    num_processes = 10
    num_jobs = 30
    random_seed = 0
    random.seed(random_seed) 

    pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)

    argument_list = [random.randint(0, 100) for _ in range(num_jobs)]

    jobs = [pool.apply_async(func, args=(i,n,)) for i, n in enumerate(argument_list)]
    pool.close()
    result_list = [job.get() for job in jobs]

    # Important to print these blanks
    print("\n" * (len(argument_list) + 1))

if __name__ == "__main__":

    main()