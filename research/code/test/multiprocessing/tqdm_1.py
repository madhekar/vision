import time
import random
from multiprocessing import Pool
from tqdm import tqdm

def my_func(a):
    time.sleep(random.random())
    return a ** 2

pool = Pool(2)
p_bar = tqdm(total=100)

def update(*a):
    p_bar.update()

for i in range(p_bar.total):
    pool.apply_async(my_func, args=(i,), callback=update)
pool.close()
pool.join()