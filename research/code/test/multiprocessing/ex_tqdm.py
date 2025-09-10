import time
import multiprocessing as mp
from multiprocessing import Pool
from random import randint

from tqdm import tqdm


def _foo(my_number):
    square = my_number * my_number
    #print('-->', square)
    time.sleep(randint(1, 2) / 2)
    return square


if __name__ == "__main__":
    max_ = 300
    ret = []
    with Pool(processes=mp.cpu_count() ) as p, tqdm(total=max_, desc='example using tqdm in pool') as pbar:
        for result in p.imap(_foo, range(0, max_,2)):
            pbar.update(2)
            pbar.refresh()
            ret.append(result)
            # do something with `result`
    p.close()
    p.join()        

    print(ret)