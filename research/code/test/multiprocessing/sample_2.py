import concurrent.futures
import multiprocessing
import time

def task_1(n):
    time.sleep(3)
    return n ** n

if __name__=='__main__':
    nums = [1,2,3,4,5,6,7,8,9,10]
    cores = multiprocessing.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        res = executor.map(task_1, nums)

        for r in res:
            print(r)