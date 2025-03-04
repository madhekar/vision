import multiprocessing
import time
import os

def f(x):
    print("PID: %d" % os.getpid())
    time.sleep(x)
    complex_obj = 5 #more complex axtually
    return complex_obj

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(4, maxtasksperchild=3)
    pool.map(f, [5]*30)
    pool.close()